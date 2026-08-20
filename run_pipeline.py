import pandas as pd
from eda_1 import load_and_clean_data
from labels_2 import engineer_labels
from featuring_3 import engineer_features
from  train_4 import split_train_test
from scaling_5 import prepare_data_for_modeling
from modalA_6 import train_and_evaluate_model_a
from  modalB_7 import train_and_evaluate_model_b
from thresh_8 import analyze_feature_importance
from  explainability_9 import evaluate_and_save_final_model
from crossdomain_10 import run_ai4i_validation

if __name__ == "__main__":
    print("--- Starting Pipeline ---")
    
    # Step 1: Load and clean data
    df_clean = load_and_clean_data('train_FD003.txt')

    # Step 2: Engineer RUL and binary labels
    df_labeled = engineer_labels(df_clean, n_threshold=20)

    # Step 3: Engineer rolling features
    df_featured = engineer_features(df_labeled)

    # Step 4: Grouped Train/Test Split (prevents engine leakage)
    train_df, test_df = split_train_test(df_featured, test_size=0.2, random_state=42)

    # Step 5: Scaling decisions and data preparation for models
    data_packets = prepare_data_for_modeling(train_df, test_df)

    # Step 6: Train and evaluate Model A (Binary Failure Prediction)
    trained_models, best_model_name = train_and_evaluate_model_a(data_packets, train_df)

    # Step 7: Model B (Root-Cause Classification)
    model_b = train_and_evaluate_model_b(train_df, test_df)

    # Step 8: Model Interpretation & Feature Importance
    best_model = trained_models[best_model_name]
    feature_cols = data_packets['feature_cols']
    feat_imp_df = analyze_feature_importance(best_model, feature_cols, top_n=10)

    # Step 9: Final Model Evaluation & Saving Artifacts
    evaluate_and_save_final_model(best_model, best_model_name, data_packets, data_packets['y_test'])

    # --- Cross-Domain Validation Pipeline (Step 10 - Bonus Domain) ---
    print("\n>>> Executing Cross-Domain Validation (AI4I 2020 Dataset)...")
    # Change path if you stored it in a subfolder like 'data/raw/ai4i2020.csv'
    ai4i_model = run_ai4i_validation('ai4i2020.csv')
    
    if ai4i_model is not None:
        print(">>> Cross-Domain Validation Completed Successfully!")
    else:
        print(">>> Skipped AI4I validation (dataset file not found locally).")

    print("\nPipeline execution test successful!")
    print(df_labeled[['unit_id', 'cycle', 'RUL', 'binary_label']].head())
    print(f"Features ready for Tree Models: {data_packets['X_train_trees'].shape}")
    print(f"Features ready for Linear Models: {data_packets['X_train_linear'].shape}")
    print(f"\nPipeline up to Task 6 Completed Successfully!")
    print(f"\nPipeline up to Task 7 Completed Successfully!")
    print(f"\nPipeline up to Task 8 Completed Successfully!")
    print(f"\n ALL PIPELINE STEPS (1 TO 10) COMPLETED SUCCESSFULLY! ")




    # --- BONUS: Live Inference Demo (Human-Readable Output) ---
    print("\n" + "="*50)
    print(" 🔍 RUNNING LIVE INFERENCE DEMO ON A TEST ENGINE... ")
    print("="*50)
    
    # ============================================================
    # 👉 CHANGE THESE TO INSPECT ANY ENGINE AT ANY POINT IN ITS LIFE
    ENGINE_ID = 5
    CYCLE = '165'# set to an integer (e.g. 30) to check an early/healthy
                        # cycle, or leave as 'latest' to check its final recorded
                        # reading (which will always be near failure by design)
    # ============================================================

    valid_ids = sorted(df_featured['unit_id'].unique())
    if ENGINE_ID not in valid_ids:
        raise ValueError(
            f"ENGINE_ID={ENGINE_ID} not found. Valid unit_ids range from "
            f"{min(valid_ids)} to {max(valid_ids)}."
        )

    engine_rows = df_featured[df_featured['unit_id'] == ENGINE_ID].sort_values('cycle')

    # Be tolerant of CYCLE being typed as a string number (e.g. '30') instead
    # of an int (30) — only the literal word 'latest' should stay a string.
    if isinstance(CYCLE, str) and CYCLE.strip().lower() != 'latest':
        CYCLE = int(CYCLE.strip())

    if isinstance(CYCLE, str) and CYCLE.strip().lower() == 'latest':
        latest_row = engine_rows.iloc[[-1]]
    else:
        min_c, max_c = int(engine_rows['cycle'].min()), int(engine_rows['cycle'].max())
        if CYCLE not in engine_rows['cycle'].values:
            raise ValueError(
                f"CYCLE={CYCLE} not found for engine {ENGINE_ID}. "
                f"Valid cycles for this engine range from {min_c} to {max_c}."
            )
        latest_row = engine_rows[engine_rows['cycle'] == CYCLE]

    raw_feature_row = latest_row[feature_cols]
    actual_status = latest_row['binary_label'].iloc[0]
    actual_rul = latest_row['RUL'].iloc[0]

    # Use the SAME representation (scaled vs raw) the best model was trained on.
    # Mirrors the logic in step9.py — feeding raw values into a model trained on
    # StandardScaler output silently saturates predict_proba to ~0 or ~1.
    if 'Trees' in best_model_name:
        sample_engine_row = raw_feature_row
    else:
        scaler = data_packets['scaler']
        sample_engine_row = pd.DataFrame(
            scaler.transform(raw_feature_row),
            columns=feature_cols,
            index=raw_feature_row.index
        )

    # Predict using the best model
    pred_prob = best_model.predict_proba(sample_engine_row)[:, 1][0]
    pred_label = best_model.predict(sample_engine_row)[0]
    
    print(f"📋 Engine ID: {ENGINE_ID} (cycle {int(latest_row['cycle'].iloc[0])}, actual RUL: {int(actual_rul)})")
    print(f"📋 Actual Status: {'Imminent Failure' if actual_status == 1 else 'Healthy'}")
    print(f"📊 Model Predicted Failure Probability: {pred_prob * 100:.2f}%")
    
    if pred_label == 1 or pred_prob > 0.5:
        print("🚨 STATUS: CRITICAL WARNING — IMMINENT FAILURE PREDICTED")
        print("   Recommended Action: Schedule immediate maintenance inspection.")
    else:
        print("✅ STATUS: SYSTEM HEALTHY / NORMAL OPERATION")
        print("   Recommended Action: Continue standard monitoring.")
        
    print("="*50)
    print(f"\n🎉 ALL PIPELINE STEPS (1 TO 10) & LIVE DEMO COMPLETED SUCCESSFULLY! 🎉")