```python
# DATA UNDERSTANDING
# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```


```python
# LOADING DATA
import pandas as pd
df = pd.read_csv("Transjakarta.csv")

# Cek Dataset
print(df.head())
```

              transID         payCardID payCardBank      payCardName payCardSex  \
    0  EIIW227B8L34VB   180062659848800      emoney   Bajragin Usada          M   
    1  LGXO740D2N47GZ  4885331907664776         dki     Gandi Widodo          F   
    2  DJWR385V2U57TO  4996225095064169         dki    Emong Wastuti          F   
    3  JTUZ800U7C86EH      639099174703       flazz     Surya Wacana          F   
    4  VMLO535V7F95NJ      570928206772       flazz  Embuh Mardhiyah          M   
    
       payCardBirthDate corridorID                              corridorName  \
    0              2008          5                     Matraman Baru - Ancol   
    1              1997         6C  Stasiun Tebet - Karet via Patra Kuningan   
    2              1992        R1A                        Pantai Maju - Kota   
    3              1978        11D       Pulo Gebang - Pulo Gadung 2 via PIK   
    4              1982         12                     Tanjung Priok - Pluit   
    
       direction tapInStops  ... tapInStopsLon  stopStartSeq            tapInTime  \
    0        1.0     P00142  ...     106.84402             7  2023-04-03 05:21:44   
    1        0.0    B01963P  ...     106.83302            13  2023-04-03 05:42:44   
    2        0.0    B00499P  ...     106.81435            38  2023-04-03 05:59:06   
    3        0.0    B05587P  ...     106.93526            23  2023-04-03 05:44:51   
    4        0.0     P00239  ...     106.88900             5  2023-04-03 06:17:35   
    
       tapOutStops        tapOutStopsName tapOutStopsLat tapOutStopsLon  \
    0       P00253                Tegalan      -6.203101      106.85715   
    1      B03307P    Sampoerna Strategic      -6.217152      106.81892   
    2      B04962P  Simpang Kunir Kemukus      -6.133731      106.81475   
    3      B03090P      Raya Penggilingan      -6.183068      106.93194   
    4       P00098       Kali Besar Barat      -6.135355      106.81143   
    
       stopEndSeq           tapOutTime  payAmount  
    0        12.0  2023-04-03 06:00:53     3500.0  
    1        21.0  2023-04-03 06:40:01     3500.0  
    2        39.0  2023-04-03 06:50:55     3500.0  
    3        29.0  2023-04-03 06:28:16     3500.0  
    4        15.0  2023-04-03 06:57:03     3500.0  
    
    [5 rows x 22 columns]
    


```python
# MENDESKRIPSIKAN DATA
df.info()
df.describe(include="all")
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 37900 entries, 0 to 37899
    Data columns (total 22 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   transID           37900 non-null  object 
     1   payCardID         37900 non-null  int64  
     2   payCardBank       37900 non-null  object 
     3   payCardName       37900 non-null  object 
     4   payCardSex        37900 non-null  object 
     5   payCardBirthDate  37900 non-null  int64  
     6   corridorID        36643 non-null  object 
     7   corridorName      35970 non-null  object 
     8   direction         37900 non-null  float64
     9   tapInStops        36687 non-null  object 
     10  tapInStopsName    37900 non-null  object 
     11  tapInStopsLat     37900 non-null  float64
     12  tapInStopsLon     37900 non-null  float64
     13  stopStartSeq      37900 non-null  int64  
     14  tapInTime         37900 non-null  object 
     15  tapOutStops       35611 non-null  object 
     16  tapOutStopsName   36556 non-null  object 
     17  tapOutStopsLat    36556 non-null  float64
     18  tapOutStopsLon    36556 non-null  float64
     19  stopEndSeq        36556 non-null  float64
     20  tapOutTime        36556 non-null  object 
     21  payAmount         36893 non-null  float64
    dtypes: float64(7), int64(3), object(12)
    memory usage: 6.4+ MB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transID</th>
      <th>payCardID</th>
      <th>payCardBank</th>
      <th>payCardName</th>
      <th>payCardSex</th>
      <th>payCardBirthDate</th>
      <th>corridorID</th>
      <th>corridorName</th>
      <th>direction</th>
      <th>tapInStops</th>
      <th>...</th>
      <th>tapInStopsLon</th>
      <th>stopStartSeq</th>
      <th>tapInTime</th>
      <th>tapOutStops</th>
      <th>tapOutStopsName</th>
      <th>tapOutStopsLat</th>
      <th>tapOutStopsLon</th>
      <th>stopEndSeq</th>
      <th>tapOutTime</th>
      <th>payAmount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>37900</td>
      <td>3.790000e+04</td>
      <td>37900</td>
      <td>37900</td>
      <td>37900</td>
      <td>37900.000000</td>
      <td>36643</td>
      <td>35970</td>
      <td>37900.000000</td>
      <td>36687</td>
      <td>...</td>
      <td>37900.000000</td>
      <td>37900.000000</td>
      <td>37900</td>
      <td>35611</td>
      <td>36556</td>
      <td>36556.000000</td>
      <td>36556.000000</td>
      <td>36556.000000</td>
      <td>36556</td>
      <td>36893.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>37900</td>
      <td>NaN</td>
      <td>6</td>
      <td>1993</td>
      <td>2</td>
      <td>NaN</td>
      <td>221</td>
      <td>216</td>
      <td>NaN</td>
      <td>2570</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37079</td>
      <td>2230</td>
      <td>2248</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35908</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>FMZZ963S4B68ZP</td>
      <td>NaN</td>
      <td>dki</td>
      <td>Suci Wacana</td>
      <td>F</td>
      <td>NaN</td>
      <td>1T</td>
      <td>Cibubur - Balai Kota</td>
      <td>NaN</td>
      <td>P00170</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-04-06 17:35:40</td>
      <td>P00016</td>
      <td>BKN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-04-24 06:53:50</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>NaN</td>
      <td>18743</td>
      <td>80</td>
      <td>20157</td>
      <td>NaN</td>
      <td>400</td>
      <td>391</td>
      <td>NaN</td>
      <td>236</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>306</td>
      <td>316</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>4.250060e+17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1990.089314</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.500633</td>
      <td>NaN</td>
      <td>...</td>
      <td>106.841554</td>
      <td>13.572480</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-6.214651</td>
      <td>106.841233</td>
      <td>21.219909</td>
      <td>NaN</td>
      <td>2699.712683</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>1.321699e+18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.051482</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.500006</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.060369</td>
      <td>12.237623</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.059022</td>
      <td>0.060999</td>
      <td>13.800689</td>
      <td>NaN</td>
      <td>4212.225592</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>6.040368e+10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1946.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>106.614730</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-6.394973</td>
      <td>106.614730</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>1.800442e+14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1982.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>106.803470</td>
      <td>4.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-6.247225</td>
      <td>106.801750</td>
      <td>11.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>3.507947e+15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1990.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>106.834830</td>
      <td>10.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-6.214718</td>
      <td>106.834580</td>
      <td>18.000000</td>
      <td>NaN</td>
      <td>3500.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>4.699023e+15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2001.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>106.882270</td>
      <td>19.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-6.174736</td>
      <td>106.883030</td>
      <td>29.000000</td>
      <td>NaN</td>
      <td>3500.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>4.997694e+18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>107.023950</td>
      <td>68.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-6.091746</td>
      <td>107.023660</td>
      <td>77.000000</td>
      <td>NaN</td>
      <td>20000.000000</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 22 columns</p>
</div>




```python
df.isnull().sum()
df = df.dropna()
```


```python
# Menambah Kolom Jumlah Transaksi
df["jumlah_transaksi"] = 1
```


```python
# DATA CLEANING UNTUK MISSING VALUES
# Cek Tipe Data Masing-Masing Kolom

import pandas as pd
df = pd.read_csv("Transjakarta.csv")
print(df.dtypes)

# Kolom numeric isi NaN dengan 0 
num_cols = df.select_dtypes(include=["int64","float64"]).columns
df[num_cols] = df[num_cols].fillna(0)

# Kolom string atau object isi NaN dengan unknown
obj_cols = df.select_dtypes(include=["object"]).columns
df[obj_cols] = df[obj_cols].fillna("unknown")

```

    transID              object
    payCardID             int64
    payCardBank          object
    payCardName          object
    payCardSex           object
    payCardBirthDate      int64
    corridorID           object
    corridorName         object
    direction           float64
    tapInStops           object
    tapInStopsName       object
    tapInStopsLat       float64
    tapInStopsLon       float64
    stopStartSeq          int64
    tapInTime            object
    tapOutStops          object
    tapOutStopsName      object
    tapOutStopsLat      float64
    tapOutStopsLon      float64
    stopEndSeq          float64
    tapOutTime           object
    payAmount           float64
    dtype: object
    


```python
# Hapus Duplikat
df.drop_duplicates(inplace=True)
```


```python
# 1. EDA
import matplotlib.pyplot as plt

# 1.1 Distribusi transaksi per jam

# Ubah ke datetime
df["tapInTime"] = pd.to_datetime(df["tapInTime"], errors="coerce")

# Buat kolom jam dari tapInTime dan ubah ke datetime
df["jam"] = df["tapInTime"].dt.hour

# Cek hasil
print(df[["tapInTime", "jam"]].head())
print(df["tapInTime"].dtype)

df.groupby("jam")["transID"].count().plot(kind="bar", figsize=(12,5))
plt.title("Distribusi Transaksi per Jam")
plt.xlabel("Jam")
plt.ylabel("Jumlah Transaksi")
plt.show()
```

                tapInTime  jam
    0 2023-04-03 05:21:44    5
    1 2023-04-03 05:42:44    5
    2 2023-04-03 05:59:06    5
    3 2023-04-03 05:44:51    5
    4 2023-04-03 06:17:35    6
    datetime64[ns]
    


    
![png](output_7_1.png)
    



```python
# 1.2 Distribusi transaksi per hari
df["hari"] = df["tapInTime"].dt.day_name()
df.groupby("hari")["transID"].count().plot(kind="bar", figsize=(10,5))
plt.title("Distribusi Transaksi per Hari")
plt.ylabel("Jumlah Transaksi")
plt.show()
```


    
![png](output_8_0.png)
    



```python
# 1.3 Menentukan Halte dengan Transaksi Tertinggi
top_halte_df = top_halte.reset_index()
top_halte_df.columns = ["halte", "jumlah transaksi"]

# Plot
plt.figure(figsize=(12,6))
sns.barplot(
    data=top_halte_df, 
    x="jumlah transaksi", 
    y="halte", 
    hue="halte",
    palette="viridis",
    dodge=False,
    legend=False
)
plt.title("Top 10 Halte dengan Transaksi Tertinggi (Tap In)")
plt.xlabel("Jumlah Transaksi")
plt.ylabel("Halte")
plt.show()
```


    
![png](output_9_0.png)
    



```python
# 1.4 Koridor dengan Transaksi Tertinggi
df["jumlah_transaksi"] = 1

koridor_df = (
    df.groupby("corridorName")["jumlah_transaksi"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    .head(10)
)

# Plot dengan seaborn
plt.figure(figsize=(10, 6))
sns.barplot(
    data=koridor_df, 
    y="corridorName", 
    x="jumlah_transaksi", 
    hue="corridorName",
    dodge=False,
    palette="coolwarm",
    legend=False
)
plt.title("Top 10 Koridor dengan Transaksi Tertinggi")
plt.ylabel("Koridor")
plt.xlabel("Jumlah Transaksi")
plt.show()
```


    
![png](output_10_0.png)
    



```python
# 1.5 Halte dengan Jumlah Tap In Terbanyak
top_halte = df["tapInStops"].value_counts().head(10)

plt.figure(figsize=(12,6))
top_halte.plot(kind="bar")
plt.title("Top 10 Halte dengan Jumlah Tap In Terbanyak")
plt.xlabel("Halte")
```




    Text(0.5, 0, 'Halte')




    
![png](output_11_1.png)
    



```python
# 1.6 Heatmap Transaksi per Jam dan Hari
# Tambahkan kolom jam & hari
df["jam"] = df["tapInTime"].dt.hour
df["hari"] = df["tapInTime"].dt.day_name()

# Menambahkan kolom jumlah_transaksi
df["jumlah_transaksi"] = 1

# Supaya nama hari urut
urutan_hari = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
df["hari"] = pd.Categorical(df["hari"], categories=urutan_hari, ordered=True)

pivot = df.pivot_table(
    values="jumlah_transaksi",
    index="hari",
    columns="jam",
    aggfunc="sum",
    fill_value=0,
    observed=False
)

plt.figure(figsize=(15,6))
sns.heatmap(pivot, cmap="YlGnBu")
plt.title("Heatmap Transaksi per Jam dan Hari")
plt.xlabel("jam")
plt.ylabel("Hari")
plt.show()
    
```


    
![png](output_12_0.png)
    



```python
# 1.7 Boxplot per jam
plt.figure(figsize=(12,6))
df.boxplot(column="jumlah_transaksi", by="jam")
plt.title("Boxplot Jumlah Transaksi per Jam")
plt.suptitle("")
plt.xlabel("Jam")
plt.ylabel("Jumlah Transaksi")
plt.show()
```


    <Figure size 1200x600 with 0 Axes>



    
![png](output_13_1.png)
    



```python
# 2. ANALISIS STATISTIK

# 2.1 INDEPENDENT T-TEST
# Membandingkan Rata-Rata Jumlah Transaksi di Weekend dan Weekday

import scipy.stats as stats

# Menambahkan Kolom Hari dan Weekend
df["hari"] = df["tapInTime"].dt.day_name()
df["jumlah_transaksi"] = 1
df["is_weekend"] = df["hari"].isin(["Saturday", "Sunday"])

# Hitung Total Transaksi per Hari
daily = df.groupby(["hari", "is_weekend"])["jumlah_transaksi"].sum().reset_index()

# Memisahkan Weekend dan Weekday
weekday = daily[daily["is_weekend"]==False]["jumlah_transaksi"]
weekend = daily[daily["is_weekend"]==True]["jumlah_transaksi"]

# T-test
t_stat, p_val = stats.ttest_ind(weekday, weekend, equal_var=False)
print("T-test Rata-rata Transaksi Weekday vs Weekend")
print("t_stat =", t_stat, "p-value =", p_val)

```

    T-test Rata-rata Transaksi Weekday vs Weekend
    t_stat = 235.0969521519112 p-value = 0.0021681303491472184
    


```python
# 2.2 ANOVA
# Menguji apakah rata-rata transaksi berbeda signifikan antar koridor

# Hitung total transaksi per koridor per hari
koridor_daily = df.groupby(["corridorName", df["tapInTime"].dt.date])["jumlah_transaksi"].sum().reset_index()

# ANOVA top 5 koridor
top5 = (df.groupby("corridorName")["jumlah_transaksi"]
        .sum().sort_values(ascending=False).head(5).index)

anova_data = [koridor_daily[koridor_daily["corridorName"]==k]["jumlah_transaksi"] for k in top5]

f_stat, p_val = stats.f_oneway(*anova_data)
print("ANOVA Perbedaan Transaksi antar Koridor (Top 5)")
print("F-stat =", f_stat, "p-value =", p_val)

# ANOVA per jam
jam_data = df.groupby([df["tapInTime"].dt.date, df["jam"]])["jumlah_transaksi"].sum().reset_index()
anova_jam = [jam_data[jam_data["jam"]==j]["jumlah_transaksi"] for j in jam_data["jam"].unique()]

f_stat, p_val = stats.f_oneway(*anova_jam)
print("ANOVA Perbedaan Transaksi antar Jam")
print("f-stat =", f_stat, "p-value =", p_val)
```

    ANOVA Perbedaan Transaksi antar Koridor (Top 5)
    F-stat = 0.6398392973310346 p-value = 0.6349998045643674
    ANOVA Perbedaan Transaksi antar Jam
    f-stat = 24.7228624600375 p-value = 6.257025943029202e-51
    


```python
# 2.3 KORELASI
# Cek Korelasi antara jumlah transaksi per jam dengan jam itu sendiri

# Agregasi transaksi perjam
df["jam"] = df["tapInTime"].dt.hour
df["jumlah_transaksi"] = 1
jam_df = df.groupby("jam")["jumlah_transaksi"].sum().reset_index()

# Korelasi Pearson
corr, p_val = stats.pearsonr(jam_df["jam"], jam_df["jumlah_transaksi"])
print("Korelasi antara Jam dan Jumlah Transaksi")
print("r =", corr, "p-value =", p_val)

# Korelasi Spearman
corr_s, p_val_s = stats.spearmanr(jam_df["jam"], jam_df["jumlah_transaksi"])
print("\nKorelasi Spearman antara Jam dan Jumlah Transaksi")
print("r =", corr_s, "p-value =", p_val_s)
```

    Korelasi antara Jam dan Jumlah Transaksi
    r = -0.12840384132751254 p-value = 0.6233287831451603
    
    Korelasi Spearman antara Jam dan Jumlah Transaksi
    r = -0.22058823529411767 p-value = 0.3948890883363402
    


```python
import pandas as pd
df = pd.read_csv("Transjakarta.csv")
print(df.head())
df.to_csv("Transjakarta.csv", index=False, sep=",")
```

              transID         payCardID payCardBank      payCardName payCardSex  \
    0  EIIW227B8L34VB   180062659848800      emoney   Bajragin Usada          M   
    1  LGXO740D2N47GZ  4885331907664776         dki     Gandi Widodo          F   
    2  DJWR385V2U57TO  4996225095064169         dki    Emong Wastuti          F   
    3  JTUZ800U7C86EH      639099174703       flazz     Surya Wacana          F   
    4  VMLO535V7F95NJ      570928206772       flazz  Embuh Mardhiyah          M   
    
       payCardBirthDate corridorID                              corridorName  \
    0              2008          5                     Matraman Baru - Ancol   
    1              1997         6C  Stasiun Tebet - Karet via Patra Kuningan   
    2              1992        R1A                        Pantai Maju - Kota   
    3              1978        11D       Pulo Gebang - Pulo Gadung 2 via PIK   
    4              1982         12                     Tanjung Priok - Pluit   
    
       direction tapInStops  ... tapInStopsLon  stopStartSeq            tapInTime  \
    0        1.0     P00142  ...     106.84402             7  2023-04-03 05:21:44   
    1        0.0    B01963P  ...     106.83302            13  2023-04-03 05:42:44   
    2        0.0    B00499P  ...     106.81435            38  2023-04-03 05:59:06   
    3        0.0    B05587P  ...     106.93526            23  2023-04-03 05:44:51   
    4        0.0     P00239  ...     106.88900             5  2023-04-03 06:17:35   
    
       tapOutStops        tapOutStopsName tapOutStopsLat tapOutStopsLon  \
    0       P00253                Tegalan      -6.203101      106.85715   
    1      B03307P    Sampoerna Strategic      -6.217152      106.81892   
    2      B04962P  Simpang Kunir Kemukus      -6.133731      106.81475   
    3      B03090P      Raya Penggilingan      -6.183068      106.93194   
    4       P00098       Kali Besar Barat      -6.135355      106.81143   
    
       stopEndSeq           tapOutTime  payAmount  
    0        12.0  2023-04-03 06:00:53     3500.0  
    1        21.0  2023-04-03 06:40:01     3500.0  
    2        39.0  2023-04-03 06:50:55     3500.0  
    3        29.0  2023-04-03 06:28:16     3500.0  
    4        15.0  2023-04-03 06:57:03     3500.0  
    
    [5 rows x 22 columns]
    


```python
df.to_csv("C:/Users/LENOVO/Downloads/Transjakarta_clean.csv", index=False, sep=",")
```


```python

```
