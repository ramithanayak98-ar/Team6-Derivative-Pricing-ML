"""
Reference Implementations using Official Python Libraries for All 5 Required Frameworks:
  1. XGBoost (`xgboost.XGBRegressor` & `xgboost.XGBClassifier`)
  2. Keras (`tensorflow.keras.Sequential`)
  3. PyTorch with fastai (`fastai.tabular.all.tabular_learner`)
  4. PyTorch with TabNet (`pytorch_tabnet.tab_model.TabNetRegressor`)
  5. PyTorch with Lightning Flash (`flash.tabular.TabularRegressionData` & `TabularRegressor`)

These modular functions provide clean, idiomatic framework code for GPU/Colab/Jupyter
environments alongside the deterministic zero-dependency engine in `src/frameworks.py`.

Authors (Team 6):
  1. Ramitha MV (PES1UG24AM392)
  2. Gayathri Priya V (PES1UG24AM105)
  3. Gonu Guntla Hema Kowshik (PES1UG24AM106)
"""


def train_xgboost_native(X_train, y_train, X_val, y_val, X_test):
    """Framework 1: Official XGBoost API (`xgboost.XGBRegressor`)."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=250,
        learning_rate=0.06,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model.predict(X_test), model.feature_importances_


def train_keras_native(X_train, y_train, X_val, y_val, X_test):
    """Framework 2: Official TensorFlow Keras Sequential MLP API."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks

    tf.random.set_seed(42)
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.10),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0,
    )
    return model.predict(X_test, verbose=0).flatten()


def train_fastai_native(df_train_val, cont_names, target_col, df_test):
    """Framework 3: Official PyTorch + fastai TabularLearner with 1-Cycle Policy."""
    from fastai.tabular.all import (
        TabularDataLoaders,
        Normalize,
        tabular_learner,
        rmse,
    )

    dls = TabularDataLoaders.from_df(
        df_train_val,
        procs=[Normalize],
        cont_names=cont_names,
        y_names=target_col,
        bs=64,
    )
    learn = tabular_learner(dls, layers=[128, 64], metrics=rmse)
    learn.fit_one_cycle(35, 3e-3)
    test_dl = learn.dls.test_dl(df_test)
    preds, _ = learn.get_preds(dl=test_dl)
    return preds.numpy().flatten()


def train_tabnet_native(X_train, y_train, X_val, y_val, X_test):
    """Framework 4: Official PyTorch + Attentive TabNet (`pytorch_tabnet`)."""
    import torch
    from pytorch_tabnet.tab_model import TabNetRegressor

    model = TabNetRegressor(
        n_d=16,
        n_a=16,
        n_steps=4,
        gamma=1.3,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="sparsemax",
        seed=42,
        verbose=0,
    )
    model.fit(
        X_train=X_train,
        y_train=y_train.reshape(-1, 1),
        eval_set=[(X_val, y_val.reshape(-1, 1))],
        eval_metric=["rmse", "mae"],
        max_epochs=80,
        patience=15,
        batch_size=128,
        virtual_batch_size=64,
    )
    preds = model.predict(X_test).flatten()
    explain_matrix, masks = model.explain(X_test)
    return preds, model.feature_importances_, explain_matrix


def train_lightning_flash_native(train_csv, val_csv, test_csv, cont_cols, target_col):
    """Framework 5: Official PyTorch + Lightning Flash TabularRegressor API."""
    import flash
    from flash.tabular import TabularRegressionData, TabularRegressor

    datamodule = TabularRegressionData.from_csv(
        num_fields=cont_cols,
        target_fields=target_col,
        train_file=train_csv,
        val_file=val_csv,
        test_file=test_csv,
        batch_size=64,
    )
    model = TabularRegressor.from_data(
        datamodule,
        backbone="tabnet",
        learning_rate=1e-2,
    )
    trainer = flash.Trainer(max_epochs=45, gpus=0)
    trainer.fit(model, datamodule=datamodule)
    preds = trainer.predict(model, datamodule=datamodule)
    return preds
