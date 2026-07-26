# EUR/USD 5 Dakikalık Fiyat Tahmini - Teknik İndikatörler ile Güncellenmiş Analiz Raporu

## 📊 Yönetici Özeti

Bu rapor, **TimeSeriesTransformer** mimarisi kullanılarak EUR/USD döviz çiftinin 5 dakikalık kapanış fiyatlarını tahmin etmek için yapılan çalışmanın teknik indikatörlerle zenginleştirilmiş versiyonunu sunmaktadır. Model, artık sadece fiyat verilerini değil, **18 teknik indikatör** ve **4 zaman özelliği** olmak üzere toplam **22 ek özellik** ile birlikte değerlendirerek daha kapsamlı bir analiz sunmaktadır.

---

## 📁 Veri Seti Özellikleri

### Temel Bilgiler
| Özellik | Değer |
|---------|-------|
| **Veri Kaynağı** | `Data/EURUSD_5m_10Yea.csv` |
| **Toplam Satır** | 748,311 |
| **Tarih Aralığı** | 17 Mayıs 2015 - 16 Mayıs 2025 (10 yıl) |
| **Zaman Periyodu** | 5 dakika |
| **Fiyat Aralığı** | 0.95390 - 1.25533 EUR/USD |
| **Ortalama Hacim** | ~974 milyon |

### Orijinal Sütunlar (OHLCV)
1. **Date** - Tarih (YYYYMMDD formatında)
2. **Time** - Saat (HH:MM:SS formatında)
3. **Open** - Açılış fiyatı
4. **High** - En yüksek fiyat
5. **Low** - En düşük fiyat
6. **Close** - Kapanış fiyatı (hedef değişken)
7. **Volume** - İşlem hacmi

---

## 🔧 Eklenen Teknik İndikatörler (18 Adet)

### 1. Trend Göstergeleri (6)
| İndikatör | Açıklama | Periyot |
|-----------|----------|---------|
| **SMA_10** | Basit Hareketli Ortalama | 10 periyot (50 dk) |
| **SMA_20** | Basit Hareketli Ortalama | 20 periyot (100 dk) |
| **SMA_50** | Basit Hareketli Ortalama | 50 periyot (250 dk) |
| **EMA_9** | Üstel Hareketli Ortalama | 9 periyot (45 dk) |
| **EMA_12** | Üstel Hareketli Ortalama | 12 periyot (60 dk) |
| **EMA_26** | Üstel Hareketli Ortalama | 26 periyot (130 dk) |

### 2. MACD Alt Göstergeleri (3)
| İndikatör | Açıklama | Hesaplama |
|-----------|----------|-----------|
| **MACD** | Moving Average Convergence Divergence | EMA_12 - EMA_26 |
| **MACD_signal** | MACD Sinyal Çizgisi | MACD'nin 9 periyotluk EMA'sı |
| **MACD_hist** | MACD Histogram | MACD - MACD_signal |

### 3. Momentum Göstergeleri (4)
| İndikatör | Açıklama | Periyot | Aralık |
|-----------|----------|---------|--------|
| **RSI_14** | Relative Strength Index | 14 periyot | 0-100 |
| **Stoch_K** | Stochastic Oscillator %K | 14 periyot | 0-100 |
| **Stoch_D** | Stochastic Oscillator %D | 3 periyotluk SMA | 0-100 |
| **Williams_R** | Williams %R | 14 periyot | -100-0 |

### 4. Volatilite Göstergeleri (5)
| İndikatör | Açıklama | Hesaplama |
|-----------|----------|-----------|
| **BB_middle** | Bollinger Bands Orta Bant | SMA_20 |
| **BB_upper** | Bollinger Bands Üst Bant | SMA_20 + 2×StdDev |
| **BB_lower** | Bollinger Bands Alt Bant | SMA_20 - 2×StdDev |
| **BB_width** | Bollinger Bant Genişliği | (Üst-Alt) / Orta |
| **BB_pct** | Bollinger Bant Pozisyonu | (Close-Alt) / (Üst-Alt) |
| **ATR_14** | Average True Range | 14 periyot |

### 5. Hacim Göstergeleri (3)
| İndikatör | Açıklama | Hesaplama |
|-----------|----------|-----------|
| **Volume_SMA_20** | Hacim Hareketli Ortalaması | 20 periyot |
| **Volume_ratio** | Hacim Oranı | Volume / Volume_SMA_20 |
| **OBV** | On-Balance Volume | Kümulatif hacim akışı |

### 6. Fiyat Değişim Özellikleri (4)
| İndikatör | Açıklama | Hesaplama |
|-----------|----------|-----------|
| **Return_1** | 1 Periyot Getiri | (Close_t - Close_t-1) / Close_t-1 |
| **Return_5** | 5 Periyot Getiri | (Close_t - Close_t-5) / Close_t-5 |
| **HL_range** | High-Low Aralığı | (High - Low) / Close |
| **OC_range** | Open-Close Aralığı | (Close - Open) / Open |

---

## 🕐 Zaman Özellikleri (4 Adet)

| Özellik | Açıklama | Aralık |
|---------|----------|--------|
| **hour** | Saat bilgisi | 0-23 |
| **day_of_week** | Haftanın günü | 0-6 (Pazartesi-Pazar) |
| **day_of_month** | Ayın günü | 1-31 |
| **month** | Ay bilgisi | 1-12 |

---

## 📈 Toplam Özellik Matriksi

| Kategori | Özellik Sayısı |
|----------|----------------|
| OHLCV (Orijinal) | 5 |
| Trend Göstergeleri | 6 |
| MACD Alt Göstergeleri | 3 |
| Momentum Göstergeleri | 4 |
| Volatilite Göstergeleri | 6 |
| Hacim Göstergeleri | 3 |
| Fiyat Değişimleri | 4 |
| Zaman Özellikleri | 4 |
| **TOPLAM** | **35** |

---

## ⚙️ Veri Ön İşleme Pipeline'ı

### 1. Eksik Veri Temizleme
- İlk 50 satır, indikatör hesaplamaları nedeniyle NaN içerir
- **Kullanılabilir net satır sayısı**: ~748,261
- Tüm NaN değerler drop edilir

### 2. Kronolojik Bölünme (Train/Val/Test)
```
Training Set:   %60  (~448,957 satır) - Model eğitimi
Validation Set: %20  (~149,652 satır) - Hiperparametre tuning
Test Set:       %20  (~149,652 satır) - Final değerlendirme
```

### 3. Özellik Ölçeklendirme
- **Target (Close)**: StandardScaler (mean=0, std=1)
- **Diğer özellikler**: MinMaxScaler veya RobustScaler
- Ölçekleyiciler kaydedilir (`target_scaler.pkl`)

### 4. Lag Features & Sequence Oluşturma
```python
CONTEXT_LENGTH = 30  # Son 30 periyot (2.5 saat)
PREDICTION_LENGTH = 1  # 1 adım ileri tahmin
LAGS_SEQUENCE = [1, 2, 3, 4, 5, 6, 7]
```

---

## 🎯 Model Konfigürasyonu (Güncellenmiş)

```json
{
  "input_features": 35,
  "context_length": 30,
  "prediction_length": 1,
  "d_model": 32,
  "encoder_layers": 2,
  "decoder_layers": 2,
  "encoder_attention_heads": 4,
  "decoder_attention_heads": 4,
  "dropout": 0.1,
  "distribution_output": "student_t",
  "learning_rate": 1e-4,
  "batch_size": 64,
  "epochs": 50,
  "early_stopping_patience": 10
}
```

**Beklenen Model Boyutu**: ~250-300 KB (özellik sayısındaki artış nedeniyle)

---

## 📊 Beklenen Performans Metrikleri

### Önceki Model (Sadece Fiyat) vs Yeni Model (İndikatörlü)

| Metrik | Önceki Model | Yeni Model (Beklenen) | İyileşme |
|--------|--------------|----------------------|----------|
| **MAE (pips)** | 2.03 | 1.70-1.85 | %8-12 |
| **RMSE (pips)** | 3.18 | 2.70-2.90 | %9-15 |
| **Directional Accuracy** | %52-54 | %55-58 | %3-4 |

### Neden Daha İyi Performans Bekleniyor?
1. **Trend bilgisi**: SMA/EMA serileri trend yönünü yakalar
2. **Momentum**: RSI, Stochastic aşırı alım/satım bölgelerini gösterir
3. **Volatilite**: Bollinger Bands ve ATR risk seviyesini belirtir
4. **Hacim analizi**: OBV ve volume ratio piyasa katılımını ölçer
5. **Zaman desenleri**: Saat/gün/ay bazlı mevsimsellik etkileri

---

## 🔬 Teknik Detaylar

### İndikatör Hesaplama Formülleri

#### RSI (Relative Strength Index)
```python
RS = Average Gain / Average Loss (14 periyot)
RSI = 100 - (100 / (1 + RS))
```

#### MACD
```python
MACD = EMA_12 - EMA_26
Signal = EMA_9(MACD)
Histogram = MACD - Signal
```

#### Bollinger Bands
```python
Middle = SMA_20
Upper = Middle + 2 × StdDev(20)
Lower = Middle - 2 × StdDev(20)
Width = (Upper - Lower) / Middle
Position = (Close - Lower) / (Upper - Lower)
```

#### ATR (Average True Range)
```python
TR = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
ATR = SMA_14(TR)
```

---

## ⚠️ Önemli Uyarılar

1. **Look-ahead Bias**: İndikatörler sadece geçmiş veri kullanır, gelecek bilgi sızıntısı yok
2. **Overfitting Riski**: 35 özellik ile model karmaşıklığı arttı, dropout ve early stopping kritik
3. **Veri Kalitesi**: İlk 50 satır silindi, toplam verinin %0.007'si kayıp
4. **Hesaplama Maliyeti**: İndikatör hesaplama O(n) zaman karmaşıklığı
5. **Trading Uyarısı**: Bu model eğitim amaçlıdır, canlı trading için uygun değildir

---

## 📦 Çıktı Artifact'leri

```
Models/
├── best_transformer_model_indicators.pth  (~280 KB)
├── target_scaler.pkl                       (879 B)
├── feature_scalers.pkl                     (~3 KB) - Tüm özellikler için
└── model_config.json                       (~1.2 KB) - Güncellenmiş config
```

---

## 🚀 Sonraki Adımlar

1. ✅ Notebook'u teknik indikatörlerle güncelle
2. ⏳ Modeli yeniden eğit (GPU önerilir)
3. ⏳ Test setinde değerlendir
4. ⏳ Feature importance analizi yap
5. ⏳ SHAP değerleri ile yorumlanabilirlik sağla
6. ⏳ Canlı veri stream'i için pipeline oluştur

---

## 📝 Sonuç

Bu güncelleme ile model, sadece ham fiyat verilerine değil, profesyonel trader'ların kullandığı teknik analiz araçlarına da erişmektedir. **35 özellikli zengin veri seti**, Transformer mimarisinin attention mekanizması ile birleştiğinde, piyasa dinamiklerini daha iyi yakalayarak tahmin doğruluğunu artırması beklenmektedir.

**Proje Durumu**: Teknik indikatör entegrasyonu tamamlandı, model eğitimi hazır.

---

*Rapor Tarihi: 2025*  
*Veri Periyodu: Mayıs 2015 - Mayıs 2025*  
*Model: TimeSeriesTransformer (Hugging Face)*
