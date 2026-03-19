use std::collections::HashSet;

use jsonrpc_core::{Error, Result};
use jsonrpc_derive::rpc;
use litesvm::types::FailedTransactionMetadata;
use serde::{Deserialize, Serialize};
use solana_client::rpc_custom_error::RpcCustomError;
use solana_signature::Signature;
use solana_transaction::versioned::VersionedTransaction;

use super::RunloopContext;
use crate::rpc::utils::decode_and_deserialize;

/// Maximum number of transactions allowed in a single bundle (matches Jito-Solana)
const MAX_TRANSACTIONS_PER_BUNDLE: usize = 5;

/// Result of simulating a single transaction within a bundle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleTransactionResult {
    /// Error message if the transaction failed, None if succeeded
    pub err: Option<String>,
    /// Transaction logs
    pub logs: Option<Vec<String>>,
    /// Compute units consumed
    pub units_consumed: Option<u64>,
}

/// Result of simulating an entire bundle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcSimulateBundleResult {
    /// "succeeded" or "failed: <reason>"
    pub summary: String,
    /// Per-transaction simulation results
    pub transaction_results: Vec<BundleTransactionResult>,
}

/// Jito-specific RPC methods for bundle submission and simulation
#[rpc]
pub trait Jito {
    type Metadata;

    /// Sends a bundle of transactions to be processed atomically.
    ///
    /// All transactions are first executed on a cloned SVM for validation.
    /// Only if ALL transactions succeed are they committed to the real SVM.
    /// If any transaction fails, the entire bundle is rejected with no state changes.
    ///
    /// ## Constraints
    /// - Bundle must contain 1-5 transactions (matches Jito-Solana `MAX_PACKETS_PER_BUNDLE`)
    /// - No duplicate transactions allowed within a bundle
    ///
    /// ## Returns
    /// A bundle ID (SHA-256 hash of comma-separated transaction signatures)
    #[rpc(meta, name = "sendBundle")]
    fn send_bundle(
        &self,
        meta: Self::Metadata,
        transactions: Vec<String>,
        config: Option<SendBundleConfig>,
    ) -> Result<String>;

    /// Simulates a bundle of transactions without committing any state changes.
    ///
    /// Transactions are executed sequentially on a cloned SVM, with each transaction
    /// seeing the state changes from previous transactions in the bundle (chain-state propagation).
    /// If any transaction fails, simulation stops and returns results up to that point.
    ///
    /// ## Returns
    /// `RpcSimulateBundleResult` with per-transaction results and overall summary
    #[rpc(meta, name = "simulateBundle")]
    fn simulate_bundle(
        &self,
        meta: Self::Metadata,
        transactions: Vec<String>,
        config: Option<SimulateBundleConfig>,
    ) -> Result<RpcSimulateBundleResult>;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SendBundleConfig {
    /// Transaction encoding (default: base58)
    pub encoding: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimulateBundleConfig {
    /// Skip signature verification (default: true for simulation)
    #[serde(default = "default_true")]
    pub skip_sig_verify: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Clone)]
pub struct SurfpoolJitoRpc;

/// Decode a list of encoded transaction strings into VersionedTransactions.
/// Returns an error if any transaction fails to decode or if there are duplicates.
fn decode_and_validate_bundle(transactions: &[String]) -> Result<Vec<VersionedTransaction>> {
    if transactions.is_empty() {
        return Err(Error::invalid_params("Bundle cannot be empty"));
    }
    if transactions.len() > MAX_TRANSACTIONS_PER_BUNDLE {
        return Err(Error::invalid_params(format!(
            "Bundle cannot exceed {} transactions, got {}",
            MAX_TRANSACTIONS_PER_BUNDLE,
            transactions.len()
        )));
    }

    let mut decoded_txs = Vec::with_capacity(transactions.len());
    let mut seen_signatures = HashSet::new();

    for (idx, tx_data) in transactions.iter().enumerate() {
        // Try base58 first, then base64
        let (_, tx) = decode_and_deserialize::<VersionedTransaction>(
            tx_data.clone(),
            solana_transaction_status::TransactionBinaryEncoding::Base58,
        )
        .or_else(|_| {
            decode_and_deserialize::<VersionedTransaction>(
                tx_data.clone(),
                solana_transaction_status::TransactionBinaryEncoding::Base64,
            )
        })
        .map_err(|e| {
            Error::invalid_params(format!("Failed to decode transaction {}: {}", idx, e))
        })?;

        // Check for duplicate transactions
        let sig = tx.signatures[0];
        if !seen_signatures.insert(sig) {
            return Err(Error::invalid_params(format!(
                "Bundle contains duplicate transaction at index {}",
                idx
            )));
        }

        decoded_txs.push(tx);
    }

    Ok(decoded_txs)
}

/// Calculate bundle ID by hashing comma-separated signatures (Jito-compatible)
/// https://github.com/jito-foundation/jito-solana/blob/master/sdk/src/bundle/mod.rs#L21
fn calculate_bundle_id(signatures: &[Signature]) -> String {
    use sha2::{Digest, Sha256};
    let concatenated = signatures
        .iter()
        .map(|sig| sig.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let mut hasher = Sha256::new();
    hasher.update(concatenated.as_bytes());
    hex::encode(hasher.finalize())
}

impl Jito for SurfpoolJitoRpc {
    type Metadata = Option<RunloopContext>;

    fn send_bundle(
        &self,
        meta: Self::Metadata,
        transactions: Vec<String>,
        _config: Option<SendBundleConfig>,
    ) -> Result<String> {
        let Some(ctx) = &meta else {
            return Err(RpcCustomError::NodeUnhealthy {
                num_slots_behind: None,
            }
            .into());
        };

        // 1. Decode and validate all transactions
        let decoded_txs = decode_and_validate_bundle(&transactions)?;
        let signatures: Vec<Signature> = decoded_txs.iter().map(|tx| tx.signatures[0]).collect();

        // 2. Dry run on cloned SVM (atomic validation)
        //    Uses clone_for_profiling() which wraps storage in overlay — no state leaks
        let dry_run_results: Vec<std::result::Result<(), FailedTransactionMetadata>> =
            ctx.svm_locker.with_svm_reader(|svm_reader| {
                let mut svm_clone = svm_reader.clone_for_profiling();
                let mut results = Vec::with_capacity(decoded_txs.len());

                for tx in &decoded_txs {
                    // Use send_transaction on clone to get chain-state propagation
                    match svm_clone.inner.send_transaction(tx.clone()) {
                        Ok(_meta) => results.push(Ok(())),
                        Err(e) => {
                            results.push(Err(e));
                            break; // Stop at first failure
                        }
                    }
                }
                results
            });

        // 3. Check if dry run succeeded for ALL transactions
        for (idx, result) in dry_run_results.iter().enumerate() {
            if let Err(e) = result {
                return Err(Error {
                    code: jsonrpc_core::ErrorCode::ServerError(-32000),
                    message: format!("Bundle rejected: transaction {} failed: {}", idx, e.err),
                    data: Some(serde_json::json!({
                        "transaction_index": idx,
                        "error": e.err.to_string(),
                        "logs": e.meta.logs,
                    })),
                });
            }
        }

        // 4. All passed — now execute on the real SVM
        ctx.svm_locker.with_svm_writer(|svm_writer| {
            for (idx, tx) in decoded_txs.iter().enumerate() {
                if let Err(e) = svm_writer.inner.send_transaction(tx.clone()) {
                    // This should not happen since dry run passed, but handle gracefully
                    return Err(Error {
                        code: jsonrpc_core::ErrorCode::ServerError(-32001),
                        message: format!(
                            "Bundle commit failed unexpectedly at transaction {}: {}",
                            idx, e.err
                        ),
                        data: None,
                    });
                }
            }
            Ok(())
        })?;

        // 5. Return bundle ID
        Ok(calculate_bundle_id(&signatures))
    }

    fn simulate_bundle(
        &self,
        meta: Self::Metadata,
        transactions: Vec<String>,
        config: Option<SimulateBundleConfig>,
    ) -> Result<RpcSimulateBundleResult> {
        let Some(ctx) = &meta else {
            return Err(RpcCustomError::NodeUnhealthy {
                num_slots_behind: None,
            }
            .into());
        };

        let _config = config.unwrap_or_default();

        // 1. Decode and validate
        let decoded_txs = decode_and_validate_bundle(&transactions)?;

        // 2. Simulate on cloned SVM (no state changes to real SVM)
        let (summary, transaction_results) = ctx.svm_locker.with_svm_reader(|svm_reader| {
            let mut svm_clone = svm_reader.clone_for_profiling();
            let mut results = Vec::with_capacity(decoded_txs.len());
            let mut all_succeeded = true;

            for tx in &decoded_txs {
                // Use send_transaction on clone to get chain-state propagation
                match svm_clone.inner.send_transaction(tx.clone()) {
                    Ok(meta) => {
                        results.push(BundleTransactionResult {
                            err: None,
                            logs: Some(meta.logs),
                            units_consumed: Some(meta.compute_units_consumed),
                        });
                    }
                    Err(e) => {
                        results.push(BundleTransactionResult {
                            err: Some(e.err.to_string()),
                            logs: Some(e.meta.logs),
                            units_consumed: Some(e.meta.compute_units_consumed),
                        });
                        all_succeeded = false;
                        break; // Stop at first failure
                    }
                }
            }

            let summary = if all_succeeded {
                "succeeded".to_string()
            } else {
                format!("failed: transaction {} failed", results.len() - 1)
            };

            (summary, results)
        });

        Ok(RpcSimulateBundleResult {
            summary,
            transaction_results,
        })
    }
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};
    use solana_keypair::Keypair;
    use solana_message::{v0::Message as V0Message, VersionedMessage};
    use solana_pubkey::Pubkey;
    use solana_signer::Signer;
    use solana_system_interface::instruction as system_instruction;
    use solana_transaction::versioned::VersionedTransaction;

    use super::*;
    use crate::tests::helpers::TestSetup;

    const LAMPORTS_PER_SOL: u64 = 1_000_000_000;

    fn build_transfer_tx(
        payer: &Keypair,
        recipient: &Pubkey,
        lamports: u64,
        recent_blockhash: &solana_hash::Hash,
    ) -> VersionedTransaction {
        let msg = VersionedMessage::V0(
            V0Message::try_compile(
                &payer.pubkey(),
                &[system_instruction::transfer(
                    &payer.pubkey(),
                    recipient,
                    lamports,
                )],
                &[],
                *recent_blockhash,
            )
            .unwrap(),
        );
        VersionedTransaction::try_new(msg, &[payer]).unwrap()
    }

    fn encode_tx(tx: &VersionedTransaction) -> String {
        bs58::encode(bincode::serialize(tx).unwrap()).into_string()
    }

    fn setup_with_funded_payer(
        lamports: u64,
    ) -> (TestSetup<SurfpoolJitoRpc>, Keypair, solana_hash::Hash) {
        let setup = TestSetup::new(SurfpoolJitoRpc);
        let payer = Keypair::new();
        let recent_blockhash = setup
            .context
            .svm_locker
            .with_svm_reader(|r| r.latest_blockhash());

        // Airdrop to payer
        setup.context.svm_locker.with_svm_writer(|w| {
            let _ = w.inner.airdrop(&payer.pubkey(), lamports);
        });

        (setup, payer, recent_blockhash)
    }

    fn get_balance(setup: &TestSetup<SurfpoolJitoRpc>, pubkey: &Pubkey) -> u64 {
        setup
            .context
            .svm_locker
            .with_svm_reader(|r| r.inner.get_account_no_db(pubkey).map_or(0, |a| a.lamports))
    }

    // =========================================================================
    // Validation Tests (based on jito-solana test_bundle_too_large, test_bundle_empty,
    //   test_bundle_duplicate_hashes)
    // =========================================================================

    #[test]
    fn test_send_bundle_empty_bundle_rejected() {
        let setup = TestSetup::new(SurfpoolJitoRpc);
        let result = setup.rpc.send_bundle(Some(setup.context), vec![], None);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("cannot be empty"),);
    }

    #[test]
    fn test_send_bundle_no_context_returns_unhealthy() {
        let setup = TestSetup::new(SurfpoolJitoRpc);
        let result = setup.rpc.send_bundle(None, vec!["tx".into()], None);
        assert!(result.is_err());
    }

    /// Based on jito-solana test_bundle_too_large:
    /// >5 transactions must be rejected
    #[test]
    fn test_send_bundle_exceeds_max_transactions() {
        let (setup, payer, blockhash) = setup_with_funded_payer(100 * LAMPORTS_PER_SOL);

        let txs: Vec<String> = (0..6)
            .map(|_| {
                let tx = build_transfer_tx(&payer, &Pubkey::new_unique(), 1000, &blockhash);
                encode_tx(&tx)
            })
            .collect();

        let result = setup.rpc.send_bundle(Some(setup.context), txs, None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.message.contains("cannot exceed 5"),
            "Expected max transaction error, got: {}",
            err.message
        );
    }

    /// Based on jito-solana test_bundle_duplicate_hashes:
    /// Same transaction appearing twice must be rejected
    #[test]
    fn test_send_bundle_duplicate_transactions() {
        let (setup, payer, blockhash) = setup_with_funded_payer(10 * LAMPORTS_PER_SOL);

        let tx = build_transfer_tx(&payer, &Pubkey::new_unique(), 1000, &blockhash);
        let encoded = encode_tx(&tx);

        let result =
            setup
                .rpc
                .send_bundle(Some(setup.context), vec![encoded.clone(), encoded], None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.message.contains("duplicate"),
            "Expected duplicate error, got: {}",
            err.message
        );
    }

    // =========================================================================
    // Atomic Execution Tests (based on jito-solana test_partial_revert_bundle,
    //   test_multi_tx_bundle_last_tx_bad_not_committed)
    // =========================================================================

    /// Based on jito-solana test_partial_revert_bundle:
    /// 2 good transfers + 1 bad (no funds) → entire bundle reverted, no state change
    #[test]
    fn test_send_bundle_atomic_rollback_on_failure() {
        let (setup, payer, blockhash) = setup_with_funded_payer(10 * LAMPORTS_PER_SOL);

        let recipient1 = Pubkey::new_unique();
        let recipient2 = Pubkey::new_unique();
        let unfunded_signer = Keypair::new(); // has no funds

        let tx1 = build_transfer_tx(&payer, &recipient1, LAMPORTS_PER_SOL, &blockhash);
        let tx2 = build_transfer_tx(&payer, &recipient2, LAMPORTS_PER_SOL, &blockhash);
        // tx3: unfunded signer tries to send — will fail
        let tx3 = build_transfer_tx(&unfunded_signer, &recipient1, LAMPORTS_PER_SOL, &blockhash);

        let payer_balance_before = get_balance(&setup, &payer.pubkey());

        let result = setup.rpc.send_bundle(
            Some(setup.context.clone()),
            vec![encode_tx(&tx1), encode_tx(&tx2), encode_tx(&tx3)],
            None,
        );

        // Bundle must fail
        assert!(result.is_err(), "Bundle with bad tx should fail");

        // Verify NO state change — payer balance unchanged
        let payer_balance_after = get_balance(&setup, &payer.pubkey());
        assert_eq!(
            payer_balance_before, payer_balance_after,
            "Payer balance should be unchanged after failed bundle (atomic rollback)"
        );

        // Recipients should have received nothing
        let r1_balance = get_balance(&setup, &recipient1);
        let r2_balance = get_balance(&setup, &recipient2);
        assert_eq!(r1_balance, 0, "Recipient1 should have 0 after rollback");
        assert_eq!(r2_balance, 0, "Recipient2 should have 0 after rollback");
    }

    /// Based on jito-solana test_multi_bundle_seed_fee_payer_ok:
    /// Chain state propagation: mint seeds kp1, kp1 sends to kp2, kp2 sends to kp3
    #[test]
    fn test_send_bundle_chain_state_propagation() {
        let (setup, payer, blockhash) = setup_with_funded_payer(10 * LAMPORTS_PER_SOL);

        let kp1 = Keypair::new();
        let kp2 = Keypair::new();
        let recipient = Pubkey::new_unique();

        // tx1: payer seeds kp1 with funds
        let tx1 = build_transfer_tx(&payer, &kp1.pubkey(), 3 * LAMPORTS_PER_SOL, &blockhash);
        // tx2: kp1 sends to kp2 (depends on tx1 state)
        let tx2 = build_transfer_tx(&kp1, &kp2.pubkey(), 2 * LAMPORTS_PER_SOL, &blockhash);
        // tx3: kp2 sends to recipient (depends on tx2 state)
        let tx3 = build_transfer_tx(&kp2, &recipient, LAMPORTS_PER_SOL, &blockhash);

        let result = setup.rpc.send_bundle(
            Some(setup.context.clone()),
            vec![encode_tx(&tx1), encode_tx(&tx2), encode_tx(&tx3)],
            None,
        );

        assert!(
            result.is_ok(),
            "Chain-state bundle should succeed: {:?}",
            result
        );

        // Verify final recipient got the funds
        let recipient_balance = get_balance(&setup, &recipient);
        assert_eq!(
            recipient_balance, LAMPORTS_PER_SOL,
            "Recipient should have received 1 SOL through chain propagation"
        );
    }

    /// Single valid transaction bundle — happy path
    #[test]
    fn test_send_bundle_single_transaction() {
        let (setup, payer, blockhash) = setup_with_funded_payer(5 * LAMPORTS_PER_SOL);
        let recipient = Pubkey::new_unique();

        let tx = build_transfer_tx(&payer, &recipient, LAMPORTS_PER_SOL, &blockhash);
        let expected_sig = tx.signatures[0];

        let result = setup
            .rpc
            .send_bundle(Some(setup.context.clone()), vec![encode_tx(&tx)], None);

        assert!(
            result.is_ok(),
            "Single tx bundle should succeed: {:?}",
            result
        );

        // Verify bundle ID matches expected SHA-256
        let bundle_id = result.unwrap();
        let mut hasher = Sha256::new();
        hasher.update(expected_sig.to_string().as_bytes());
        let expected_id = hex::encode(hasher.finalize());
        assert_eq!(bundle_id, expected_id);

        // Verify state changed
        let balance = get_balance(&setup, &recipient);
        assert_eq!(balance, LAMPORTS_PER_SOL);
    }

    // =========================================================================
    // simulateBundle Tests
    // =========================================================================

    /// Simulate a valid 2-tx bundle — should succeed with no state changes
    #[test]
    fn test_simulate_bundle_success_no_state_change() {
        let (setup, payer, blockhash) = setup_with_funded_payer(10 * LAMPORTS_PER_SOL);
        let recipient1 = Pubkey::new_unique();
        let recipient2 = Pubkey::new_unique();

        let tx1 = build_transfer_tx(&payer, &recipient1, LAMPORTS_PER_SOL, &blockhash);
        let tx2 = build_transfer_tx(&payer, &recipient2, LAMPORTS_PER_SOL, &blockhash);

        let payer_balance_before = get_balance(&setup, &payer.pubkey());

        let result = setup.rpc.simulate_bundle(
            Some(setup.context.clone()),
            vec![encode_tx(&tx1), encode_tx(&tx2)],
            None,
        );

        assert!(result.is_ok());
        let sim_result = result.unwrap();
        assert_eq!(sim_result.summary, "succeeded");
        assert_eq!(sim_result.transaction_results.len(), 2);
        assert!(sim_result.transaction_results[0].err.is_none());
        assert!(sim_result.transaction_results[1].err.is_none());

        // Verify NO state change on real SVM
        let payer_balance_after = get_balance(&setup, &payer.pubkey());
        assert_eq!(
            payer_balance_before, payer_balance_after,
            "Simulate should NOT change real SVM state"
        );
        let r1_balance = get_balance(&setup, &recipient1);
        assert_eq!(r1_balance, 0, "Recipient1 should have 0 after simulate");
        let r2_balance = get_balance(&setup, &recipient2);
        assert_eq!(r2_balance, 0, "Recipient2 should have 0 after simulate");
    }

    /// Based on jito-solana test_multi_tx_bundle_last_tx_bad_not_committed:
    /// Simulate 3 good txs + 1 bad — returns partial results, stops at failure
    #[test]
    fn test_simulate_bundle_failure_mid_bundle() {
        let (setup, payer, blockhash) = setup_with_funded_payer(10 * LAMPORTS_PER_SOL);
        let unfunded = Keypair::new();

        let tx1 = build_transfer_tx(&payer, &Pubkey::new_unique(), 1000, &blockhash);
        let tx2 = build_transfer_tx(&payer, &Pubkey::new_unique(), 1000, &blockhash);
        let tx3 = build_transfer_tx(&unfunded, &Pubkey::new_unique(), 1000, &blockhash); // bad

        let result = setup.rpc.simulate_bundle(
            Some(setup.context.clone()),
            vec![encode_tx(&tx1), encode_tx(&tx2), encode_tx(&tx3)],
            None,
        );

        assert!(result.is_ok());
        let sim_result = result.unwrap();
        assert!(sim_result.summary.starts_with("failed"));
        assert_eq!(
            sim_result.transaction_results.len(),
            3,
            "Should have results for 2 good + 1 failed"
        );
        assert!(
            sim_result.transaction_results[0].err.is_none(),
            "tx1 should succeed"
        );
        assert!(
            sim_result.transaction_results[1].err.is_none(),
            "tx2 should succeed"
        );
        assert!(
            sim_result.transaction_results[2].err.is_some(),
            "tx3 should fail"
        );
    }

    /// Simulate chain-state propagation in bundle
    #[test]
    fn test_simulate_bundle_chain_state_propagation() {
        let (setup, payer, blockhash) = setup_with_funded_payer(10 * LAMPORTS_PER_SOL);

        let kp1 = Keypair::new();
        let recipient = Pubkey::new_unique();

        // tx1: payer seeds kp1
        let tx1 = build_transfer_tx(&payer, &kp1.pubkey(), 3 * LAMPORTS_PER_SOL, &blockhash);
        // tx2: kp1 sends to recipient (depends on tx1)
        let tx2 = build_transfer_tx(&kp1, &recipient, LAMPORTS_PER_SOL, &blockhash);

        let result = setup.rpc.simulate_bundle(
            Some(setup.context.clone()),
            vec![encode_tx(&tx1), encode_tx(&tx2)],
            None,
        );

        assert!(result.is_ok());
        let sim_result = result.unwrap();
        assert_eq!(sim_result.summary, "succeeded");
        assert_eq!(sim_result.transaction_results.len(), 2);
        assert!(sim_result.transaction_results[0].err.is_none());
        assert!(sim_result.transaction_results[1].err.is_none());

        // State unchanged on real SVM
        let kp1_balance = get_balance(&setup, &kp1.pubkey());
        assert_eq!(
            kp1_balance, 0,
            "kp1 should have 0 on real SVM after simulate"
        );
    }

    /// simulateBundle validation: empty bundle
    #[test]
    fn test_simulate_bundle_empty_rejected() {
        let setup = TestSetup::new(SurfpoolJitoRpc);
        let result = setup.rpc.simulate_bundle(Some(setup.context), vec![], None);
        assert!(result.is_err());
    }

    /// simulateBundle validation: >5 txs
    #[test]
    fn test_simulate_bundle_exceeds_max_rejected() {
        let (setup, payer, blockhash) = setup_with_funded_payer(100 * LAMPORTS_PER_SOL);
        let txs: Vec<String> = (0..6)
            .map(|_| {
                encode_tx(&build_transfer_tx(
                    &payer,
                    &Pubkey::new_unique(),
                    1000,
                    &blockhash,
                ))
            })
            .collect();

        let result = setup.rpc.simulate_bundle(Some(setup.context), txs, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("cannot exceed 5"));
    }
}
