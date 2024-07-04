import pandas as pd

inputs = """
Prompt:
Summarize the text following  'Text to Summarize'  Focusing only on the following aspects:

Aspects:
1) Legacy and Recognition: if no info for an aspect -> do not output anything 
2) Influence on Spanish Football Tactics and Style: if no info for an aspect -> do not output anything
3) Death 

Format: For each aspect a bullet, according to the above aspects
Text to Summarize:    
== حرفه باشگاهی ==
او که در [[A Coruña]] به دنیا آمد، در جوانی به [[مادرید]] منتقل شد و در مدرسه Nuestra Senora del Pilar که زمانی یکی از مهدهای فوتبال شهر بود شروع به تمرین فوتبال کرد و از آنجا وارد شد. دسته های تمرینی [[رئال مادرید CF|رئال مادرید]] در پایان فصل 1918-1918.<ref name=vida>{{cite web |url=http://hemeroteca.abc.es/nav/Navigate .exe/hemeroteca/madrid/abc/1950/11/15/019.html |title=El famoso ex-jugador de fútbol Monjardín muere víctima de un aksidente de automóvil |trans-title=منژاردین بازیکن معروف سابق فوتبال درگذشت قربانی یک تصادف رانندگی |language=es |publisher=Diario ABC|accessdate=10 ژوئن 2022}}</ref> در همان فصل، و علیرغم سن 15 سالگی اش، اولین بار با تیم اول در [[ Campeonato Regional Centro|Central Regional Championship]] در برابر [[Racing de Madrid]]، و او به سرعت به یکی از معیارهای باشگاه در آن زمان تبدیل شد. او به زودی از پست خود به عنوان هافبک به مهاجم تبدیل شد که دیگر تا پایان دوران حرفه ای خود از آن دست نکشید.<ref name=monjardin>{{cite web|url=https://www.marca.com/blogs/ni -mas-ni-menos/2015/03/21/juanito-monjardin-el-de-las-piernas.html |title=Juanito Monjardín, el de las piernas torcidas |trans-title=Juanito Monjardín، کسی که پاهای کج دارد |language=es |publisher=Diario Marca|author=Jesús Ramos |accessdate=10 ژوئن 2021}}</ref>

بین قهرمانی منطقه ای فوق الذکر و کوپا دل ری، او در مجموع 55 گل در 74 بازی به ثمر رساند. در زمان بازنشستگی در سال 1929 (26 سالگی)، او دومین گلزن برتر تیم مادریدی بود که تنها با 68 گل هم تیمی اش [[سانتیاگو برنابئو یسته|سانتیاگو برنابئو]] از او پیشی گرفت.{{نیست منبع|date= ژوئن 2022}} یکی از دلایل بازنشستگی زودهنگام او آمدن دو بازیکن بود که در نهایت به عنوان مرجع هجومی و تاریخی باشگاه، بازیکن والنسیا [[گاسپار روبیو]] و [[خائمه لازکانو]] از [[ناوارا] بودند. ]]، هر دو جوانتر از او، و هر دو در نهایت رکورد گلزنی او را در باشگاه شکستند. در همان فصل دوران بازنشستگی او در لالیگا افتتاح شد و با انجام تنها یک بازی که آن هم تنها بازی او در آن فصل بود، به یکی از 19 بازیکن باشگاهی تبدیل شد که در اولین دوره تاریخی حضور داشتند.<ref>{{ cite web|url=http://www.bdfutbol.com/es/t/t1928-292.html|title=Plantilla Real Madrid 1928-29|publisher=Portal digital BDFutbol|accessdate=10 ژوئن 2022}}</ref >

در سال 1943، سال‌ها پس از بازنشستگی حرفه‌ای او، باشگاه سفید به افتخار او مسابقه‌ای را بین مردم مادرید و [[FC Barcelona| بارسلونا]] برگزار کرد که با تساوی یک گل به پایان رسید.<ref>{{cite web| url=http://hemeroteca.mundodeportivo.com/preview/1943/10/23/pagina-2/651251/pdf.html|title=''رئال مادرید، 1 - بارسلونا، 1''(PDF)|ناشر= Diario El Mundo Deportivo|accessdate=10 ژوئن 2022}}</ref>

==حرفه بین المللی==
او که یک بازیکن [[رئال مادرید سی اف سی|ف سی مادرید]] بود، واجد شرایط بازی برای [[تیم فوتبال خودمختار مادرید|'Centro' (تیم نماینده منطقه مادرید)]] بود، و او بخشی از تیمی بود که شرکت کرد. در دو تورنمنت [[جام شاهزاده آستوریاس]]، یک رقابت بین منطقه ای، در [[جام شاهزاده آستوریاس 1922–23|1922–1923]] و [[1923–24 جام شاهزاده آستوریاس|1923–24 ]]، و اگرچه اولین بازی با یک خروج تکان دهنده از یک چهارم نهایی به دست [[تیم ملی فوتبال گالیسیا|گالیسیا]] به پایان رسید، که در آن مونژاردین گل تسلی مادرید را در باخت 1–4 به ثمر رساند، دومین دوره بازی بسیار زیاد بود. بهتر است، تا حد زیادی به لطف مونژاردین که دو بار در برد 2-1 آنها مقابل [[تیم ملی فوتبال اندلس|اندلس یازدهم]] در نیمه نهایی گلزنی کرد و پس از آن چیزی که به نظر می رسید یک [[اضافی (ورزشی)# بود. فدراسیون فوتبال|وقت اضافه]] برنده در مقابل [[تیم ملی فوتبال کاتالونیا|کاتالونیا]] در [[فینال جام شاهزاده آستوریاس | فینال]] برای دومین قهرمانی مادرید در جام شاهزاده آستوریاس، اما گل تساوی در آخرین لحظه از [[امیلی ساگی باربا]] مجبور به بازی مجدد شد که در آن او دوباره گلزنی کرد و در نیمه اول دو گل به ثمر رساند، اما تلاش های او بی فایده بود زیرا کاتالونیا با پیروزی 3-2 عنوان قهرمانی را از آن خود کرد.<ref name=Prince> {{cite web |url=http://www.cihefe.es/cuadernosdefutbol/2009/09/la-copa-principe-de-asturias/ |title=La Copa Príncipe de Asturias |trans-title=The Prince of Asturias جام |language=es |publisher=[[:es:Centro de Investigaciones de Historia y Estadística del Fútbol Español|CIHEFE]] |author=Vicente Martínez Calatrava |date=17 اوت 2009 |accessdate=5 ژوئن 2022}}</ref > خط نقره ای این است که او با پنج گل، بهترین گلزن جام شاهزاده آستوریاس در فصل 24-1923 بود و با مجموع شش گل در این رقابت ها، [[جام شاهزاده آستوریاس#تمام دوران] برترین گلزنان|برترین گلزن مشترک تاریخ]] جام شاهزاده آستوریاس به همراه [[خوزه لوئیس زابالا]] و [[خوان آرمت|کینکه]].

او اولین بازی خود را برای [[تیم ملی فوتبال اسپانیا|تیم ملی اسپانیا]] در [[لیسبون]] در 17 دسامبر 1922 مقابل [[تیم ملی فوتبال پرتغال|پرتغال]] انجام داد و گل پیروزی 2-1 را به ثمر رساند. در دقیقه 82 او در بازی بعدی خود در برابر [[تیم ملی فوتبال فرانسه|فرانسه]] در 28 ژانویه 1923، در پیروزی 3-0 یک گل به ثمر رساند و تصادفاً نویسنده گل سوم زابالا بود. در مجموع، او چهار بار بازی کرد و سه گل به ثمر رساند.<ref name=EU>{{cite web|url=https://eu-football.info/_player.php?id=14272 |title=Juan Monjardín |کار =EU-football.info |access-date=10 ژوئن 2022}}</ref>
"""

llm_resp_inputs = """
Summary:
- Juanito Monjardín, one of Real Madrid's early stars, was born in A Coruña but moved to Madrid as a youth, joining the club's youth setup. He went on to become the second-highest goalscorer in the club's history at the time of his retirement in 1929, aged only 26. His legacy is honoured by the club even today.

- Monjardín's influence on the club extended beyond his goal-scoring prowess. He transitioned from a midfield role to become a prolific striker, a position he held throughout his career. This tactical adaptability had a lasting impact on Spanish football strategy.

- Sadly, Monjardín's life was cut short in a car accident. The exact date and circumstances of his death are not specified in the text.
"""
# Create the DataFrame
data = {
    'inputs': [inputs],
    'llm_resp_inputs': [llm_resp_inputs]
}
df = pd.DataFrame(data)

# Save to Parquet file
df.to_parquet('summarization_input.parquet', engine='pyarrow')

# Display the DataFrame
print(df.info())
