# **Telecomunication Company Analysis : Customer Churn**

## **Business Problem Understanding**
### **Context**  
Sebuah perusahaan yang telah lama bergerak di industri Telekomunikasi memahami bahwa persaingan semakin ketat dalam beberapa tahun terakhir. Beberapa perusahaan pesaing melakukan merger satu sama lain ataupun mengalami akuisisi oleh perusahaan internasional dengan tambahan modal yang sangat besar. perusahaan pesaing lama namun dengan modal yang lebih besar dari sebelumnya mulai melakukan perbaikan layanan dengan memperluas dan memperkuat jaringan koneksinya. Dengan kualitas layanan yang bersaing tersebut, mereka mulai menawarkan tarif langganan yang lebih murah. Perusahaan memiliki keunggulan yaitu jaringan yang dimiliki lebih tersebar hingga ke pelosok negeri.

Perusahaan tersebut ingin memprediksi karakteristik *customer* seperti apa yang akan berhenti langganan (*churn*). Sebagai seorang *Data Scientist* kita diminta untuk membuat model *Machine Learning* yang sesuai agar jumlah *customer churn* dapat dikurangi.

Target :

0 : Tidak Berhenti Langganan

1 : Berhenti Langganan (Churn)

### **Problem Statement**

Pelanggan berhenti langganan (*churn*) merupakan masalah yang dapat menyebabkan kerugian apabila jumlahnya sangat tinggi. Masalah tersebut dapat juga dijadikan sebagai indikator bahwa perusahaan kalah dalam bersaing dengan kompetitor untuk menawarkan layanan yang optimal kepada pelanggan. 

Mendapatkan pelanggan baru akan memerlukan *cost* yang tidak sedikit, seperti *cost* untuk marketing, potongan harga dan lain-lain. Sebagai contoh biaya akuisi untuk 1 orang pelanggan dari perusahaan asal Amerika [Sprint PCS](https://www.entrepreneur.com/growing-a-business/how-much-did-that-new-customer-cost-you/225415) adalah $315. Menurut hasil survey yang dilakukan oleh website [berikut](https://www.huify.com/blog/acquisition-vs-retention-customer-lifetime-value) *cost* untuk mendapatkan pelanggan baru adalah 5 kali lebih besar daripada *cost* untuk mempertahankan pelanggan lama. Artinya, untuk mempertahankan 1 orang pelanggan hanya membutuhkan 1/5 dari biaya akuisisi yaitu $63. Oleh karena itu, perusahaan harus berfokus untuk menurunkan persentase *churn* dengan meningkatkan retensi pada pelanggan lama agar kerugian dari kehilangan pelanggan dapat diminimalisir.



### **Goals**

Maka berdasarkan permasalahan tersebut, perusahaan ingin memiliki kemampuan untuk memprediksi kemungkinan seorang pelanggan akan berhenti berlangganan (*churn*) atau tidak, sehingga perusahaan dapat memfokuskan *strategy* bisnisnya untuk menjaga retensi pelanggan.

Dan juga, perusahaan ingin mengetahui faktor/variabel apa yang membuat seorang pelanggan tetap bertahan, sehingga mereka dapat membuat rencana yang lebih baik dalam membuat program-program layanan untuk pelanggan supaya tingkat (*churn*) dapat berukurang.



### **Analytic Approach**

Jadi yang akan kita lakukan adalah menganalisis data untuk menemukan pola yang membedakan pelanggan yang berhenti langganan (*churn*) dan tidak. Kemudian kita akan membangun model klasifikasi yang akan membantu perusahaan untuk dapat memprediksi probabilitas seorang pelanggan akan pindah ke provider lain atau tidak

### **Metric Evaluation**

Target :
- 0 : Tidak berhenti berlangganan
- 1 : Berhenti berlangganan (*churn*)


Type 1 error : False Positive (pelanggan yg diprediksi *churn* namun aktual tidak *churn*)
Konsekuensi: kehilangan calon pelanggan loyal

Type 2 error : False Negative (pelanggan yg diprediksi tidak *churn* namun aktual *churn*) 
Konsekuensi: kerugiaan akibat biaya untuk mempertahankan pelanggan (retensi) tidak tepat sasaran

Sesuai dengan pernyataan problem statement sebelumnya dan beberapa sumber yang saya peroleh dari internet bahwa:
 
- Menurut sumber yang saya peroleh bahwa *customer lifetime period* berada pada kisaran [12-24 Bulan](https://www.pm.lth.se/fileadmin/pm/Exjobb/Filer_fram_till_foerra_aaret/Exjobb_2013/Flordahl___Friberg/CLV_ERICSSON_Flordal_Friberg.pdf). Menurut *common sense* sebagai pelanggan salah satu penyedia internet di Indonesia, kontrak minimum berlangganan ada pada jangka waktu 1 tahun (12 bulan). Maka dalam perhitungan nantinya saya akan menggunakan *customer lifetime period* 12 months.
- Biaya untuk *customer accuistion* pelanggan sebesar $315/12 months = $26.25/month.
- Untuk mempertahankan 1 orang pelanggan hanya membutuhkan 1/5 dari biaya *customer accuistion* yaitu $5.25/month.
- Rata-rata *monthly* charge untuk pelanggan *telco customer* yaitu [$64.76](https://www.analyticsvidhya.com/blog/2022/01/churn-analysis-of-a-telecom-company/).

Maka dapat kita simulasikan perhitungan berikut:
- Jika kita lebih memfokuskan prediksi untuk memperkecil error False Positif kita akan mencegah perusahaaan mengalami potensi kerugian sebesar $64.76 + $5.25 = $70.01/month per pelanggan.
- Jika kita lebih memfokuskan prediksi untuk memperkecil error False Negatif kita akan mencegah perusahaaan mengalami potensi kerugian sebesar $64.76 + $26.25 + $5.25 = $96.26/month per pelanggan.


Berdasarkan konsekuensinya, maka sebisa mungkin yang akan kita lakukan adalah membuat model yang dapat mengurangi customer berhenti langganan (*churn*) dari perusahaan tersebut. Meminimalisasi prediksi **False Negative** (pelanggan yg diprediksi tidak *churn* namun aktual *churn*) menjadi fokus utama kita. Maka metric utama yang akan kita gunakan adalah f beta score 2 karena kita menganggap **Recall** lebih penting daripada **Precision**. Dengan kata lain kita lebih fokus untuk memperkecil error false negatif alih-alih false positif.

## **Data Understanding**
Dataset yang kita miliki memiliki informasi sebagai berikut :

- Jumlah baris sebanyak 4930 baris
- Terdapat 10 kolom
- Type data sudah sesuai
- Pelanggan yang berhenti berlangganan (Yes|No) adan pada kolom `Churn`
- Servis atau layanan yang dimiliki customer ada pada kolom `Online Security`, `Online Backup`, `Internet Service`, `Device Protection`, `Tech Support`
- Informasi demografi mengenai apakah pelanggan memiliki tanggungan ada pada kolom `Dependants`
- Informasi akun pelanggan `Tenure`, `Contract`, `PaperlessBilling`, `MonthlyCharges`

Berikut adalah informasi data type dari setiap *attribute* beserta deskripsinya :

| Attribute | Data Type | Description |
| --- | --- | --- |
| Dependents | 'Object' | Whether the customer has dependents : Yes or No. Dependents could be children, parents, grandparents, etc. |
| tenure | Integer | Number of months the customer has stayed with the company|
| OnlineSecurity | 'Object' | Whether the customer has online security or not |
| OnlineBackup | 'Object' | Whether the customer has online backup or not |
| InternetService | 'Object' | Whether the client is subscribed to Internet service |
| DeviceProtection | 'Object' | Whether the client has device protection or not |
| TechSupport | 'Object' | Whether the client has tech support or not |
| Contract | 'Object' | Type of contract according to duration |
| PaperlessBilling | 'Object' | Bills issued in paperless form |
| MonthlyCharges | Float | Amount of charge for service on monthly bases |
| Churn | 'Object' | Yes = the customer left the company this period. No = the customer remained with the company |

#### **Conclusion**
Berdasarkan permodelan yang kita lakukan dapat diperoleh kesimpulan sebagai berikut  :
- Dampak yang disebabkan oleh False Negatif (Pelanggan yang aktual tidak **churn** diprediksi **churn**) lebih besar dibandingkan False Positif (Pelanggan yang aktual **churn** diprediksi tidak **churn**) sehingga metrics yang kita gunakan ada F beta score dengan nila beta = 2. F2 score merupakan metrics yang digunakan untuk menyeimbangkan Precision dan recall namun recall kita anggap 2 kali lebih penting.

- Pada tahap *cross validation* kita memperoleh 2 model yang paling stabil baik pada train set dan test set namun menghasilkan score terbaik, yaitu  : **Logistic Regression** dan **AdaBoost**

- Setelah melalui proses hyperparameter tuning pada model Logistic Regression dan AdaBoost diperoleh score terbaik dengan parameter sebagai berikut :
    * Logistic Regression --> 0.7474402730375426

        Parameter : 
        
            - 'C': 0.01
            - 'class_weight': None
            - 'fit_intercept': False
            - 'max_iter': 50
            - 'penalty': 'l1'
            - 'solver': 'saga'
            - 'k_neighbors': 7

    * AdaBoost --> 0.7634827810266406
    
        Parameter : 
            
            - 'algorithm': 'SAMME'
            - 'learning_rate': 0.09
            - 'model__n_estimators': 65

- Dari best model Logistic Regression diperoleh interpretasi dari koefisien regressi untuk setiap *features* sebagai berikut :

    * `InternetService` : Pelanggan dengan layanan internet (fiber optic, DSL) cenderung berhenti berlangganan.
    * `PaperlessBilling` : Penggunaan PaperlessBilling berkorelasi lemah dengan peluang pelanggan untuk berhenti berlangganan.
    * Fitur Tambahan (`OnlineBackup`, `DeviceProtection`, `MonthlyCharges`) : Tidak ada pengaruh signifikan dari fitur-fitur ini terhadap peluang berhenti berlangganan.
    * Dukungan Teknis (`TechSupport`), Keamanan Online (`OnlineSecurity`), dan Tanggungan (`Dependents`) : Pengaruh kecil hingga tidak signifikan terhadap peluang pelanggan berhenti berlangganan.
    * Lama Berlangganan (`tenure`) : Hubungan invers; semakin lama pelanggan berlangganan, semakin rendah peluang berhenti.
    * Jenis Kontrak (`Contract`) : Pelanggan dengan kontrak jangka panjang memiliki peluang lebih rendah untuk berhenti berlangganan.
    <br><br>
- Dari best model AdaBoost diperoleh *features importances* yang berpengaruh dalam pengambilan keputusan model tersebut, yaitu:

    * `Contract`: Kontrak jangka panjang memiliki kontribusi tertinggi dalam memprediksi *churn*, menunjukkan pelanggan dengan kontrak lebih lama cenderung tetap setia.
    * `InternetService` : Jenis layanan internet, terutama fiber optic dan DSL, berperan signifikan dalam memprediksi *churn*, penggunaan layanan tersebut lebih cenderung untuk *churn*.
    * `PaperlessBilling` dan `tenure` : Penggunaan PaperlessBilling dan tenure memiliki kontribusi moderat, dimana pengguna PaperlessBilling lebih cenderung untuk *churn*, sementara pelanggan dengan tenure lebih lama cenderung tetap berlangganan.
    * `Dependents` : Keberadaan tanggungan (*dependents*) memberikan kontribusi rendah dalam prediksi *churn*, menunjukkan pelanggan tanpa tanggungan lebih cenderung untuk *churn*.
    * Fitur-fitur lainnya (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `MonthlyCharges`): Fitur-fitur ini dianggap tidak penting dalam prediksi *churn* karena kontribusinya mendekati nol.

- Walaupun AdaBoost menghasilkan F2 score lebih baik, jika kita bandingkan metrics F1 Score diperoleh kesimpulan bahwa Logistic Regression lebih seimbang dalam memperkecil False Positive dan False Negatif, terlihat dari score-nya yang lebih besar dibandingkan model AdaBoost. 
   * Logistic Regression --> F1 Score = 0.633864
        - **False Positif = 214**
        - True Positif  = 235
        - **False Negatif = 39**
        - True Negatif  = 441
        <br><br>
   * AdaBoost --> F1 Score = 0.614379
        - **False Positif = 272**
        - True Positif  = 299
        - **False Negatif = 23**
        - True Negatif  = 419
    Terlihat bahwa model logistic Regression memberikan penurunan yang lebih seimbang dibandingkan AdaBoost. Nilai False Negatifnya lebih tinggi namun dapat menurunkan false negatif lebih signifikan.

**Business**

- **Cost Retention** :
   
    - Model Logistic Regression menghemat cost retention sebesar **23.79%**.
    - Model AdaBoost menghemat cost retention sebesar **20.521%**.
<br><br>
- **Pemilihan Model terhadap Dampak Finansial** :
    - Menggunakan Logistic Regression daripada AdaBoost dapat menghemat **$181.676**.
    - Walaupun F2 Score AdaBoost lebih besar, Logistic Regression lebih baik dalam menurunkan False Positif.
<br><br>
- **F1 Score dan Dampak Finansial** :
    - Logistic Regression menunjukkan keseimbangan yang lebih baik dalam memperkecil False Positive dan False Negative.
    - Dengan performa yang lebih baik, Logistic Regression membantu perusahaan menurunkan kerugian akibat biaya Retensi dan Akuisisi.

Meskipun AdaBoost memiliki F2 Score yang lebih besar, keputusan menggunakan Logistic Regression didasarkan pada analisis dampak finansial yang lebih positif dan kemampuannya dalam menurunkan False Positif.
