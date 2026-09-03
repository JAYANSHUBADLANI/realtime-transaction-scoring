# Data

The raw dataset is **not** committed to this repository (6.36M rows,
~470 MB). Download it yourself with any of the three options below, then
place the CSV in this `data/` folder as `paysim_transactions.csv`, matching
`config.yaml`'s `data.csv_path`.

## Dataset

**PaySim: Synthetic Financial Datasets For Fraud Detection**, by ntnu-testimon,
mirrored on Kaggle as `ealaxi/paysim1`.

- Kaggle URL: https://www.kaggle.com/datasets/ealaxi/paysim1
- 6,362,620 simulated mobile money transactions over 743 simulation hours
  (`step`, an hourly index, not a real calendar timestamp; see below).
- Real fraud labels (`isFraud`), confirmed by EDA to occur only within the
  `TRANSFER` and `CASH_OUT` transaction types (0 fraud in the other three
  types across all 6,354,407 non-fraud-eligible rows). This project scores
  only those two types by design; see `config.yaml`'s `data.scored_types`
  comment for why.
- Also carries `isFlaggedFraud`, PaySim's own naive built-in rule (fires on
  16 of 6.36M rows), kept in this project as a weak baseline to evaluate
  real detectors against.

## Option A: kagglehub (recommended, no credentials needed)

Confirmed working with zero Kaggle account setup on the machine this project
was built on:

```bash
pip install kagglehub
python scripts/download_data.py
```

This pulls the dataset via kagglehub's public mirror and copies it to
`data/paysim_transactions.csv` automatically.

## Option B: Kaggle API

1. Install the CLI: `pip install kaggle`
2. Create an API token at https://www.kaggle.com/settings (Account, then
   "Create New Token"). This downloads `kaggle.json`.
3. Move it into place and lock down permissions:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Download and unzip into this folder:
   ```bash
   kaggle datasets download -d ealaxi/paysim1 -p data/ --unzip
   mv data/PS_20174392719_1491204439457_log.csv data/paysim_transactions.csv
   ```

## Option C: Manual download

1. Open the Kaggle URL above and sign in.
2. Click **Download** to get the zip.
3. Unzip it and rename the CSV to `paysim_transactions.csv` in this `data/`
   folder.

## Schema

| Column | Description |
| --- | --- |
| `step` | Simulation hour index, 1 to 743 (~31 days). Not a real timestamp; the pipeline derives a synthetic `event_time` from it, see `config.yaml`. |
| `type` | `PAYMENT`, `TRANSFER`, `CASH_OUT`, `CASH_IN`, or `DEBIT`. |
| `amount` | Transaction amount, local currency units. |
| `nameOrig`, `nameDest` | Sender / recipient account IDs. |
| `oldbalanceOrg`, `newbalanceOrig` | Sender's balance before / after. |
| `oldbalanceDest`, `newbalanceDest` | Recipient's balance before / after. |
| `isFraud` | Ground truth fraud label. |
| `isFlaggedFraud` | PaySim's own built-in rule flag (weak baseline). |

`src/load_data.py` engineers three additional columns from the ledger fields
above (`balance_error_orig`, `balance_error_dest`, `orig_emptied`); see its
docstring for exactly how and why. If the file you download uses different
column names, update `config.yaml`'s `data.required_columns` to match;
`load_data.py` validates the required columns and reports any that are
missing.
