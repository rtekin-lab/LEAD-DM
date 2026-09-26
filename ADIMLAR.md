# Depo kurulum adımları

Sırayla uygulayın. Her adım bir öncekine bağlı.

---

## 1. Platform seçimi

**GitHub + Zenodo birlikte.** İkisi farklı iş yapar:

| | GitHub | Zenodo |
|---|---|---|
| Amaç | geliştirme, sürüm kontrolü | kalıcı arşiv, DOI |
| Değişebilir mi | evet | hayır, sürüm dondurulur |
| Atıf verilebilir mi | hayır | **evet, DOI ile** |

Dergiler **kalıcı tanımlayıcı** ister. Yalnız GitHub adresi vermek yetmez;
depo silinirse veya yeniden adlandırılırsa bağlantı kırılır. Zenodo, GitHub
deposunun bir anlık görüntüsünü alıp DOI üretir.

Alternatif olarak **OSF** veya **figshare** de kullanılabilir, ancak GitHub
entegrasyonu Zenodo'da en olgun olanıdır.

---

## 2. GitHub deposunu oluşturun

```bash
cd lead-dm
git init
git add .
git commit -m "LEAD-DM: initial public release"
git branch -M main
git remote add origin https://github.com/<kullanici>/lead-dm.git
git push -u origin main
```

**Depoyu ilk gönderimde gizli (private) tutmak da bir seçenektir.** Çift kör
hakemlikte açık depo kimliğinizi ele verir. İki yol var:

* Depoyu kabul edilene kadar gizli tutun, makalede "yayım üzerine
  yayımlanacaktır" deyin. En yaygın uygulama budur.
* Ya da anonim bir Zenodo/OSF kopyası oluşturup hakemlere onu verin.

---

## 3. Zenodo bağlantısı ve DOI

1. zenodo.org adresine GitHub hesabınızla girin
2. **Settings → GitHub** bölümünden `lead-dm` deposunu açık konuma getirin
3. GitHub'da bir sürüm yayımlayın:

```bash
git tag -a v1.0.0 -m "Version accompanying the published article"
git push origin v1.0.0
```

4. Zenodo otomatik olarak arşivler ve iki DOI üretir:
   * **Concept DOI** — tüm sürümleri gösterir, makalede **bunu** kullanın
   * **Version DOI** — yalnız v1.0.0

Makaledeki `[REPOSITORY URL AND DOI]` yerine şunu yazın:

> The implementation is available at https://github.com/\<kullanici\>/lead-dm
> and archived at https://doi.org/10.5281/zenodo.XXXXXXX

---

## 4. Ağırlıkları nereye koyacaksınız

Model kontrol noktaları **GitHub'a değil Zenodo'ya**. Gerekçe: gürültü
tahmincisi 11,43 M parametre, fp32'de yaklaşık 46 MB; otokodlayıcı 0,379 M.
Üç veri seti için hepsi birlikte 150 MB'ı aşar. Git bu tür ikili dosyaları
her sürümde yeniden saklar, depo hızla şişer.

Ayrı bir Zenodo kaydı açın: "LEAD-DM trained model weights". İçine:

```
AE_final_ptbxl.pt        otokodlayıcı
AE_final_cpsc2018.pt
AE_final_chapman.pt
M2_final_ptbxl.pt        difüzyon modeli
M2_final_cpsc2018.pt
M2_final_chapman.pt
config.json              her koşunun tam yapılandırması
SHA256SUMS               bütünlük doğrulaması
```

README'deki "Trained weights" bölümüne bu kaydın DOI'sini yazın.

---

## 5. Ne paylaşılır, ne paylaşılmaz

**Paylaşılır**

| Ne | Neden |
|---|---|
| Tüm `.py` kaynak dosyaları | tekrarlanabilirliğin temeli |
| `prepare_data.py` | ön işlemeyi belirleyen tek dosya |
| Şekil üretim betikleri | şekillerin veriden nasıl doğduğunu gösterir |
| `requirements.txt` | ortam |
| Yapılandırma dosyaları | hiperparametreler |
| Eğitilmiş ağırlıklar (Zenodo) | eğitimi tekrarlamadan üretim |

**Paylaşılmaz**

| Ne | Neden |
|---|---|
| PTB-XL, CPSC2018, Chapman ham verisi | sağlayıcının lisansı, yeniden dağıtım hakkı yok |
| `data/prepared/*.h5` | ham veriden türetilmiş, aynı lisansa tabi |
| Üretilmiş sentetik kayıtlar | koddan yeniden üretilebilir, gereksiz hacim |
| Kişisel yol adları, API anahtarları | `lead_dm/config.py` içindeki yerel yolları kontrol edin |

**⚠️ Göndermeden önce mutlaka kontrol edin:** `lead_dm/config.py` içinde
`D:\users\rtekin\...` gibi yerel makine yolları kalmış olabilir. Bunlar hem
çalışmaz hem de gereksiz bilgi sızdırır.

```bash
grep -rn "D:\\\\\|C:\\\\\|/home/\|/Users/" src/ figures/ scripts/
```

---

## 6. Gönderimden önceki son kontrol

```bash
# 1. Temiz bir ortamda kurulum çalışıyor mu
python -m venv /tmp/test && source /tmp/test/bin/activate
pip install -r requirements.txt

# 2. Modüller hatasız içe aktarılıyor mu
python -c "import sys; sys.path.insert(0,'src'); import lead_dm; print('ok')"

# 3. Veri veya ağırlık dosyası yanlışlıkla eklenmiş mi
git ls-files | grep -E "\.(h5|npy|pt|ckpt)$" && echo "TEMIZLE" || echo "temiz"

# 4. Depo boyutu makul mü (10 MB altı olmalı)
du -sh .git
```

---

## 7. Makalede güncellenecek yerler

| Dosya | Yer tutucu | Yazılacak |
|---|---|---|
| EN makale | `REPOSITORY-URL` | GitHub adresi + Zenodo DOI |
| TR makale | `DEPO-ADRESI` | aynısı |
| BSPC yazar beyanları | `[REPOSITORY URL AND DOI]` | aynısı |
| SREP Data availability | `[REPOSITORY URL AND DOI]` | aynısı |
| `CITATION.cff` | `<user>`, ORCID, DOI | gerçek değerler |
| `README.md` | `<user>`, ağırlık DOI'si | gerçek değerler |

---

## 8. Zamanlama önerisi

1. **Şimdi:** depoyu yerel olarak hazırlayın, gizli GitHub deposu açın
2. **Gönderimde:** makalede "yayım üzerine yayımlanacaktır" deyin
3. **Kabul sonrası:** depoyu açık yapın, sürüm etiketleyin, Zenodo DOI'sini alın
4. **Düzelti aşamasında:** DOI'yi makaleye işleyin

Bu sıra en yaygın olanıdır ve çift kör hakemlikle uyumludur.
