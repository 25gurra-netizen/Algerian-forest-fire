# Task 1 – Know Your Data: Algerian Forest Fires

## 1. Dataset overview

The uploaded file contains observations from two Algerian regions:
- Bejaia
- Sidi-Bel Abbes

After parsing the two sections and correcting one malformed row in the source file, the dataset contains **244 observations** and **15 columns**, including the region and target class.

Target variable: **Classes**
- fire: 138
- not fire: 106

Regions:
- Bejaia: 122
- Sidi-Bel Abbes: 122

## 2. Features

The dataset contains the following variables:

Region, day, month, year, Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, FWI, Classes

The numerical variables are:
day, month, year, Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, FWI

`Region` and `Classes` are categorical variables.

`Classes` is the target variable for the later classification task.

## 3. Data quality

Missing values found after parsing and correcting the malformed source row:

Region         0
day            0
month          0
year           0
Temperature    0
RH             0
Ws             0
Rain           0
FFMC           0
DMC            0
DC             0
ISI            0
BUI            0
FWI            0
Classes        0

Number of duplicate rows: **0**

The original CSV has a formatting problem in one observation: the value sequence contains `14.6 9` without a comma. This was interpreted as two separate values (`DC = 14.6` and `ISI = 9`) because the surrounding rows and the expected 14-column structure confirm this interpretation.

## 4. Data exploration and visualization

The generated plots investigate:
1. Distribution of the target classes.
2. Temperature by fire class.
3. FWI by fire class.
4. Correlations between numerical features.
5. Class distribution by region.

These plots can be included in the report to demonstrate the exploration required by Task 1.

## 5. Missing-value handling

No missing values remain after parsing the supplied dataset. Therefore, no imputation is required for this particular prepared dataset.

Nevertheless, different imputation techniques that could be used if missing values were present include:
- Mean imputation for numerical variables.
- Median imputation for numerical variables, especially when outliers are present.
- Mode imputation for categorical variables.

Because this supplied dataset contains no missing values after correction, applying imputation would not add information and was therefore not necessary.

## 6. Preprocessing decisions

- Whitespace around column names and categorical values was removed.
- Numerical attributes were converted to numerical data types.
- `Classes` was retained as the target variable.
- `Region` was retained because it describes the observation's geographical source and may be useful during exploration. Its use as a model feature can be reconsidered during Task 2.
- The source's malformed row was corrected.
- No attributes were removed solely on the basis of Task 1 exploration.

### Important consideration for Task 2

`FWI` is a fire-weather index and is strongly related to several other fire-weather variables. Since the goal in Task 2 is classification, the group should discuss whether using FWI together with its component indices is appropriate and whether any variables could cause information leakage or redundancy. This decision belongs in the Task 2 preprocessing/modeling discussion.

## 7. Result

The cleaned dataset is ready to be used as the input to Task 2, where the data will be split into training/validation/test sets and classifiers will be trained.
