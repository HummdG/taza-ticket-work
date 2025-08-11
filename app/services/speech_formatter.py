import re
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# --------------------------------------
# Public API
# --------------------------------------

def format_flight_for_speech(flight_response: str, language: str = 'en') -> str:
    """Convert a structured flight response string to natural speech."""
    return flight_speech_formatter.convert_to_natural_speech(flight_response, language)


# --------------------------------------
# Core Class
# --------------------------------------

class FlightSpeechFormatter:
    """Convert structured flight responses into natural human speech."""

    def __init__(self):
        # IATA to nice spoken names (airports & metro codes). Extend as needed.
        self.city_names: Dict[str, str] = {
            # Pakistan
            'LHE': 'Lahore', 'KHI': 'Karachi', 'ISB': 'Islamabad',
            # UK / London area
            'LHR': 'London Heathrow', 'LGW': 'London Gatwick', 'STN': 'London Stansted',
            'LTN': 'London Luton', 'LCY': 'London City Airport', 'SEN': 'London Southend', 'LON': 'London',
            # USA
            'JFK': 'New York JFK', 'LGA': 'New York LaGuardia', 'EWR': 'Newark', 'NYC': 'New York',
            'LAX': 'Los Angeles', 'SFO': 'San Francisco', 'ORD': "Chicago O'Hare", 'DFW': 'Dallas–Fort Worth',
            'MIA': 'Miami', 'SEA': 'Seattle', 'BOS': 'Boston Logan', 'IAD': 'Washington Dulles',
            'DCA': 'Washington National', 'ATL': 'Atlanta', 'PHX': 'Phoenix', 'LAS': 'Las Vegas', 'HNL': 'Honolulu',
            # Europe
            'BER': 'Berlin', 'FRA': 'Frankfurt', 'MUC': 'Munich',
            'CDG': 'Paris Charles de Gaulle', 'ORY': 'Paris Orly', 'PAR': 'Paris',
            'AMS': 'Amsterdam', 'BRU': 'Brussels', 'MAD': 'Madrid', 'BCN': 'Barcelona',
            'ATH': 'Athens', 'ZRH': 'Zurich', 'VIE': 'Vienna', 'CPH': 'Copenhagen',
            'OSL': 'Oslo', 'ARN': 'Stockholm Arlanda', 'HEL': 'Helsinki', 'DUB': 'Dublin',
            'MAN': 'Manchester', 'EDI': 'Edinburgh', 'GLA': 'Glasgow', 'LIS': 'Lisbon',
            'PRG': 'Prague', 'WAW': 'Warsaw', 'BUD': 'Budapest', 'KBP': 'Kyiv Boryspil',
            'SVO': 'Moscow Sheremetyevo', 'DME': 'Moscow Domodedovo',
            # Middle East
            'DXB': 'Dubai', 'DWC': 'Dubai World Central', 'AUH': 'Abu Dhabi', 'DOH': 'Doha',
            'JED': 'Jeddah', 'RUH': 'Riyadh', 'MCT': 'Muscat', 'AMM': 'Amman', 'BEY': 'Beirut',
            'IST': 'Istanbul', 'SAW': 'Istanbul Sabiha Gökçen', 'CAI': 'Cairo', 'SHJ': 'Sharjah',
            # India & South Asia
            'DEL': 'Delhi', 'BOM': 'Mumbai', 'MAA': 'Chennai', 'BLR': 'Bengaluru',
            'HYD': 'Hyderabad', 'COK': 'Kochi', 'DAC': 'Dhaka', 'KTM': 'Kathmandu', 'CMB': 'Colombo',
            # East & SE Asia / Oceania
            'HKG': 'Hong Kong', 'SIN': 'Singapore', 'KUL': 'Kuala Lumpur', 'BKK': 'Bangkok',
            'HND': 'Tokyo Haneda', 'NRT': 'Tokyo Narita', 'ICN': 'Seoul Incheon',
            'PVG': 'Shanghai Pudong', 'PEK': 'Beijing Capital', 'CAN': 'Guangzhou',
            'TPE': 'Taipei', 'SYD': 'Sydney', 'MEL': 'Melbourne', 'AKL': 'Auckland',
            # Africa
            'JNB': 'Johannesburg', 'CPT': 'Cape Town', 'NBO': 'Nairobi', 'ADD': 'Addis Ababa',
            'LOS': 'Lagos', 'CMN': 'Casablanca',
            # Latin America
            'GRU': 'São Paulo Guarulhos', 'GIG': 'Rio de Janeiro Galeão', 'EZE': 'Buenos Aires Ezeiza',
            'SCL': 'Santiago', 'LIM': 'Lima', 'MEX': 'Mexico City', 'BOG': 'Bogotá',
            # Canada
            'YYZ': 'Toronto Pearson', 'YVR': 'Vancouver', 'YUL': 'Montreal',
            # Misc
            'Various': 'multiple cities'
        }

        # Airline codes → spoken names
        self.airline_names: Dict[str, str] = {
            'EK': 'Emirates', 'PK': 'Pakistan International Airlines', 'QR': 'Qatar Airways',
            'EY': 'Etihad Airways', 'TK': 'Turkish Airlines', 'BA': 'British Airways',
            'LH': 'Lufthansa', 'AF': 'Air France', 'KL': 'KLM', 'SQ': 'Singapore Airlines',
            'CX': 'Cathay Pacific', 'PC': 'Pegasus Airlines', '3U': 'Sichuan Airlines',
            'MU': 'China Eastern Airlines', 'DL': 'Delta Air Lines', 'UA': 'United Airlines',
            'AA': 'American Airlines', 'AC': 'Air Canada', 'VA': 'Virgin Australia',
            'VS': 'Virgin Atlantic', 'WN': 'Southwest', 'AZ': 'ITA Airways', 'IB': 'Iberia',
            'AY': 'Finnair', 'SK': 'Scandinavian Airlines', 'LO': 'LOT Polish Airlines',
            'LY': 'El Al', 'ET': 'Ethiopian Airlines', 'KQ': 'Kenya Airways',
            'SA': 'South African Airways', 'MK': 'Air Mauritius', 'MH': 'Malaysia Airlines',
            'GA': 'Garuda Indonesia', 'JL': 'Japan Airlines', 'NH': 'ANA', 'KE': 'Korean Air',
            'OZ': 'Asiana Airlines', 'FZ': 'Flydubai', 'W6': 'Wizz Air', 'U2': 'easyJet',
            'FR': 'Ryanair', 'G9': 'Air Arabia', 'Various': 'multiple airlines'
        }

        self.currency_names = {
            'USD': 'US dollars', 'EUR': 'euros', 'GBP': 'British pounds', 'AED': 'UAE dirhams',
            'PKR': 'Pakistani rupees', 'INR': 'Indian rupees', 'SAR': 'Saudi riyals',
            'QAR': 'Qatari riyals', 'AUD': 'Australian dollars', 'CAD': 'Canadian dollars'
        }

        # Month lookup for parsing dates when regex gives e.g. "Sep 05"
        self.months_short = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

        # ---------- MULTI-LANGUAGE STRINGS ----------
        # Keys: greeting, price, airline, flight_no, leaves, lands, on, at, from, to, total_time,
        # direct, has_stops, baggage_fee, baggage_inc, baggage_generic, close
        self.LANG_STRINGS: Dict[str, Dict[str, str]] = {
            'en': {
                'greeting': "Great news — I've got a flight that fits!",
                'price': "It’s {price} {currency}.",
                'airline': "You’ll fly with {airline}.",
                'flight_no': "Flight number {flight}.",
                'leaves': "leaves",
                'lands': "lands",
                'on': "on",
                'at': "at",
                'from': "from",
                'to': "in",
                'total_time': "Total travel time is {duration}.",
                'direct': "It’s a direct flight.",
                'has_stops': "This itinerary has {stops}.",
                'baggage_fee': "Checked bag fees may apply.",
                'baggage_inc': "One checked bag is included.",
                'baggage_generic': "Baggage: {text}.",
                'close': "Want me to look for more options or start booking?"
            },
            'ur': {
                'greeting': "Zabardast! Aap ke liye ek achhi flight mil gayi hai.",
                'price': "Ticket ki qeemat {price} {currency} hai.",
                'airline': "Aap {airline} ke saath safar kareinge.",
                'flight_no': "Flight number {flight}.",
                'leaves': "ravangi",
                'lands': "pahunch",
                'on': "ko",
                'at': "par",
                'from': "se",
                'to': "mein",
                'total_time': "Total safar ka waqt {duration} hai.",
                'direct': "Yeh seedhi flight hai, beech mein koi stop nahin.",
                'has_stops': "Is flight mein {stops} shamil hain.",
                'baggage_fee': "Checked baggage ka charge lag sakta hai.",
                'baggage_inc': "Ek checked bag shamil hai.",
                'baggage_generic': "Baggage: {text}.",
                'close': "Batayein, aur options dikhaun ya booking mein madad karun?"
            },
            'hi': {  # simple Hindi (can merge with ur if you want)
                'greeting': "बहुत बढ़िया! आपके लिए एक अच्छी फ्लाइट मिल गई है.",
                'price': "टिकट की कीमत {price} {currency} है.",
                'airline': "आप {airline} के साथ यात्रा करेंगे.",
                'flight_no': "फ़्लाइट नंबर {flight}.",
                'leaves': "रवाना होती है",
                'lands': "पहुंचती है",
                'on': "को",
                'at': "पर",
                'from': "से",
                'to': "में",
                'total_time': "कुल यात्रा समय {duration} है.",
                'direct': "यह सीधी उड़ान है.",
                'has_stops': "इस उड़ान में {stops} हैं.",
                'baggage_fee': "चेक्ड बैग पर शुल्क लग सकता है.",
                'baggage_inc': "एक चेक्ड बैग शामिल है.",
                'baggage_generic': "बैगेज: {text}.",
                'close': "क्या मैं और विकल्प दिखाऊं या बुकिंग शुरू करूं?"
            },
            'ar': {
                'greeting': "خبر رائع! وجدت لك رحلة مناسبة.",
                'price': "السعر {price} {currency}.",
                'airline': "ستسافر مع شركة {airline}.",
                'flight_no': "رقم الرحلة {flight}.",
                'leaves': "تغادر",
                'lands': "تصل",
                'on': "في",
                'at': "عند الساعة",
                'from': "من",
                'to': "إلى",
                'total_time': "مدة الرحلة الإجمالية {duration}.",
                'direct': "الرحلة مباشرة بدون توقف.",
                'has_stops': "هذه الرحلة تحتوي على {stops}.",
                'baggage_fee': "قد تُطبق رسوم على الأمتعة المسجلة.",
                'baggage_inc': "تشمل حقيبة شحن واحدة.",
                'baggage_generic': "الأمتعة: {text}.",
                'close': "هل تريد أن أتابع الحجز أو أبحث عن خيارات أخرى؟"
            },
            # ---- New languages ----
            'el': {  # Greek
                'greeting': "Καλά νέα! Βρήκα μια κατάλληλη πτήση.",
                'price': "Η τιμή είναι {price} {currency}.",
                'airline': "Θα πετάξετε με την {airline}.",
                'flight_no': "Αριθμός πτήσης {flight}.",
                'leaves': "αναχωρεί",
                'lands': "προσγειώνεται",
                'on': "στις",
                'at': "στις",
                'from': "από",
                'to': "στην",
                'total_time': "Συνολικός χρόνος ταξιδιού {duration}.",
                'direct': "Είναι απευθείας πτήση.",
                'has_stops': "Η διαδρομή έχει {stops}.",
                'baggage_fee': "Ενδέχεται να ισχύουν χρεώσεις αποσκευών.",
                'baggage_inc': "Περιλαμβάνεται μία αποσκευή.",
                'baggage_generic': "Αποσκευές: {text}.",
                'close': "Θέλετε να συνεχίσω με κράτηση ή να βρω κι άλλες επιλογές;"
            },
            'it': {
                'greeting': "Ottime notizie! Ho trovato un volo adatto.",
                'price': "Costa {price} {currency}.",
                'airline': "Volerai con {airline}.",
                'flight_no': "Numero di volo {flight}.",
                'leaves': "parte",
                'lands': "arriva",
                'on': "il",
                'at': "alle",
                'from': "da",
                'to': "a",
                'total_time': "Il tempo totale di viaggio è {duration}.",
                'direct': "È un volo diretto.",
                'has_stops': "L’itinerario prevede {stops}.",
                'baggage_fee': "Potrebbero essere applicati costi per il bagaglio registrato.",
                'baggage_inc': "Un bagaglio registrato è incluso.",
                'baggage_generic': "Bagaglio: {text}.",
                'close': "Vuoi che cerchi altre opzioni o procedo con la prenotazione?"
            },
            'fr': {
                'greeting': "Bonne nouvelle ! J’ai trouvé un vol adapté.",
                'price': "Le prix est de {price} {currency}.",
                'airline': "Vous voyagerez avec {airline}.",
                'flight_no': "Numéro de vol {flight}.",
                'leaves': "décolle",
                'lands': "atterrit",
                'on': "le",
                'at': "à",
                'from': "de",
                'to': "à",
                'total_time': "La durée totale du voyage est de {duration}.",
                'direct': "C’est un vol direct.",
                'has_stops': "L’itinéraire comporte {stops}.",
                'baggage_fee': "Des frais de bagages enregistrés peuvent s’appliquer.",
                'baggage_inc': "Un bagage enregistré est inclus.",
                'baggage_generic': "Bagages : {text}.",
                'close': "Souhaitez-vous d’autres options ou commencer la réservation ?"
            },
            'de': {
                'greeting': "Gute Nachrichten – ich habe einen passenden Flug gefunden!",
                'price': "Er kostet {price} {currency}.",
                'airline': "Sie fliegen mit {airline}.",
                'flight_no': "Flugnummer {flight}.",
                'leaves': "startet",
                'lands': "landet",
                'on': "am",
                'at': "um",
                'from': "ab",
                'to': "in",
                'total_time': "Gesamtreisezeit: {duration}.",
                'direct': "Es ist ein Direktflug.",
                'has_stops': "Diese Verbindung hat {stops}.",
                'baggage_fee': "Für aufgegebenes Gepäck können Gebühren anfallen.",
                'baggage_inc': "Ein aufgegebenes Gepäckstück ist inklusive.",
                'baggage_generic': "Gepäck: {text}.",
                'close': "Möchten Sie weitere Optionen sehen oder mit der Buchung starten?"
            },
            'es': {
                'greeting': "¡Buenas noticias! Tengo un vuelo que encaja.",
                'price': "Cuesta {price} {currency}.",
                'airline': "Volarás con {airline}.",
                'flight_no': "Número de vuelo {flight}.",
                'leaves': "sale",
                'lands': "llega",
                'on': "el",
                'at': "a las",
                'from': "desde",
                'to': "a",
                'total_time': "El tiempo total de viaje es {duration}.",
                'direct': "Es un vuelo directo.",
                'has_stops': "Este itinerario tiene {stops}.",
                'baggage_fee': "Puede aplicarse un cargo por equipaje facturado.",
                'baggage_inc': "Incluye una maleta facturada.",
                'baggage_generic': "Equipaje: {text}.",
                'close': "¿Busco más opciones o empezamos a reservar?"
            },
            'nl-be': {  # Belgian Dutch
                'greeting': "Goed nieuws! Ik heb een geschikte vlucht gevonden.",
                'price': "De prijs is {price} {currency}.",
                'airline': "Je vliegt met {airline}.",
                'flight_no': "Vluchtnummer {flight}.",
                'leaves': "vertrekt",
                'lands': "landt",
                'on': "op",
                'at': "om",
                'from': "van",
                'to': "in",
                'total_time': "Totale reistijd is {duration}.",
                'direct': "Het is een rechtstreekse vlucht.",
                'has_stops': "Deze reis heeft {stops}.",
                'baggage_fee': "Er kunnen kosten voor ingecheckte bagage gelden.",
                'baggage_inc': "Eén ingecheckte koffer is inbegrepen.",
                'baggage_generic': "Bagage: {text}.",
                'close': "Wil je meer opties of zal ik beginnen met boeken?"
            },
            'ka': {  # Georgian
                'greeting': "კარგი ამბავია! იპოვე შესაფერისი რეისი.",
                'price': "ფასი არის {price} {currency}.",
                'airline': "იფრენთ {airline}-ით.",
                'flight_no': "ფრენის ნომერი {flight}.",
                'leaves': "გასვლა",
                'lands': "ჩამოსვლა",
                'on': "ზე",
                'at': "ზე",
                'from': "დან",
                'to': "ში",
                'total_time': "სრული მგზავრობის დროა {duration}.",
                'direct': "ეს არის პირდაპირი რეისი.",
                'has_stops': "ამ რეისს აქვს {stops}.",
                'baggage_fee': "შეიძლება მოქმედებდეს ჩასაბარებელი ბარგის საკომისიო.",
                'baggage_inc': "ერთი ჩასაბარებელი ბარგი შედის.",
                'baggage_generic': "ბარგი: {text}.",
                'close': "გსურთ სხვა ვარიანტები დავძებნო თუ დაჯავშნა დავიწყო?"
            },
            'bn': {  # Bengali
                'greeting': "দারুণ খবর! আপনার জন্য একটি উপযুক্ত ফ্লাইট পেয়েছি।",
                'price': "মূল্য {price} {currency}।",
                'airline': "আপনি {airline} দিয়ে ভ্রমণ করবেন।",
                'flight_no': "ফ্লাইট নম্বর {flight}।",
                'leaves': "রওনা দেবে",
                'lands': "পৌঁছাবে",
                'on': "তারিখে",
                'at': "সময়",
                'from': "থেকে",
                'to': "এ",
                'total_time': "মোট যাত্রার সময় {duration}।",
                'direct': "এটি একটি সরাসরি ফ্লাইট।",
                'has_stops': "এই ফ্লাইটে {stops} রয়েছে।",
                'baggage_fee': "চেকড ব্যাগের জন্য ফি প্রযোজ্য হতে পারে।",
                'baggage_inc': "একটি চেকড ব্যাগ অন্তর্ভুক্ত।",
                'baggage_generic': "ব্যাগেজ: {text}।",
                'close': "আরো অপশন দেখাবো নাকি বুকিং শুরু করবো?"
            },
            'zh': {  # Simplified generic
                'greeting': "好消息！我找到了一趟合适的航班。",
                'price': "价格是 {price} {currency}。",
                'airline': "您将乘坐 {airline}。",
                'flight_no': "航班号 {flight}。",
                'leaves': "起飞",
                'lands': "到达",
                'on': "于",
                'at': "在",
                'from': "从",
                'to': "到",
                'total_time': "总行程时间为 {duration}。",
                'direct': "这是直飞航班。",
                'has_stops': "该行程有 {stops}。",
                'baggage_fee': "托运行李可能需要额外费用。",
                'baggage_inc': "包含一件托运行李。",
                'baggage_generic': "行李：{text}。",
                'close': "需要我继续预订还是查看更多选项？"
            },
            'ko': {
                'greeting': "좋은 소식이에요! 딱 맞는 항공편을 찾았어요.",
                'price': "가격은 {price} {currency}입니다.",
                'airline': "{airline} 항공으로 이용하십니다.",
                'flight_no': "항공편 번호 {flight}.",
                'leaves': "출발",
                'lands': "도착",
                'on': "",
                'at': "",
                'from': "출발지",
                'to': "도착지",
                'total_time': "총 소요 시간은 {duration}입니다.",
                'direct': "직항편입니다.",
                'has_stops': "경유는 {stops} 입니다.",
                'baggage_fee': "위탁 수하물 요금이 발생할 수 있습니다.",
                'baggage_inc': "위탁 수하물 1개가 포함됩니다.",
                'baggage_generic': "수하물: {text}.",
                'close': "더 찾아볼까요, 아니면 예약을 진행할까요?"
            },
            'ja': {
                'greeting': "朗報です。条件に合うフライトが見つかりました。",
                'price': "料金は{price}{currency}です。",
                'airline': "{airline}をご利用いただきます。",
                'flight_no': "フライト番号は{flight}です。",
                'leaves': "出発",
                'lands': "到着",
                'on': "",
                'at': "",
                'from': "出発地",
                'to': "到着地",
                'total_time': "総所要時間は{duration}です。",
                'direct': "直行便です。",
                'has_stops': "経由地は{stops}です。",
                'baggage_fee': "受託手荷物には追加料金がかかる場合があります。",
                'baggage_inc': "受託手荷物1個が含まれます。",
                'baggage_generic': "手荷物：{text}。",
                'close': "他の選択肢を探しますか？それとも予約を進めますか？"
            },
            'pt': {
                'greeting': "Boa notícia! Encontrei um voo ideal.",
                'price': "O preço é {price} {currency}.",
                'airline': "Você voará com {airline}.",
                'flight_no': "Número do voo {flight}.",
                'leaves': "parte",
                'lands': "chega",
                'on': "em",
                'at': "às",
                'from': "de",
                'to': "em",
                'total_time': "O tempo total de viagem é {duration}.",
                'direct': "É um voo direto.",
                'has_stops': "Este itinerário tem {stops}.",
                'baggage_fee': "Podem ser aplicadas taxas de bagagem despachada.",
                'baggage_inc': "Uma bagagem despachada está incluída.",
                'baggage_generic': "Bagagem: {text}.",
                'close': "Quer ver mais opções ou começar a reservar?"
            },
            'ru': {
                'greeting': "Отличная новость! Я нашёл подходящий рейс.",
                'price': "Цена {price} {currency}.",
                'airline': "Вы полетите с {airline}.",
                'flight_no': "Номер рейса {flight}.",
                'leaves': "вылетает",
                'lands': "прибывает",
                'on': " ",
                'at': "в",
                'from': "из",
                'to': "в",
                'total_time': "Общее время в пути {duration}.",
                'direct': "Это прямой рейс.",
                'has_stops': "В маршруте {stops}.",
                'baggage_fee': "Может взиматься плата за зарегистрированный багаж.",
                'baggage_inc': "Одна зарегистрированная сумка включена.",
                'baggage_generic': "Багаж: {text}.",
                'close': "Показать ещё варианты или перейти к бронированию?"
            },
            'uk': {
                'greeting': "Чудова новина! Знайшов для вас відповідний рейс.",
                'price': "Вартість {price} {currency}.",
                'airline': "Ви летітимете з {airline}.",
                'flight_no': "Номер рейсу {flight}.",
                'leaves': "вилітає",
                'lands': "прибуває",
                'on': "",
                'at': "о",
                'from': "з",
                'to': "до",
                'total_time': "Загальний час у дорозі {duration}.",
                'direct': "Це прямий рейс.",
                'has_stops': "Маршрут має {stops}.",
                'baggage_fee': "Може стягуватися плата за багаж.",
                'baggage_inc': "Одна одиниця багажу включена.",
                'baggage_generic': "Багаж: {text}.",
                'close': "Показати більше варіантів чи перейти до бронювання?"
            },
            'sr': {  # Serbian (Latin)
                'greeting': "Sjajne vesti! Pronašao sam odgovarajući let.",
                'price': "Cena je {price} {currency}.",
                'airline': "Letećete sa kompanijom {airline}.",
                'flight_no': "Broj leta {flight}.",
                'leaves': "polazi",
                'lands': "sleće",
                'on': "",
                'at': "u",
                'from': "iz",
                'to': "u",
                'total_time': "Ukupno trajanje putovanja je {duration}.",
                'direct': "Let je direktan.",
                'has_stops': "Ovaj itinerer ima {stops}.",
                'baggage_fee': "Može se naplatiti taksa za predati prtljag.",
                'baggage_inc': "Jedan predati kofer je uključen.",
                'baggage_generic': "Prtljag: {text}.",
                'close': "Da potražim još opcija ili da krenemo sa rezervacijom?"
            }
        }

        # Aliases mapping
        self.lang_aliases = {
            'be': 'nl-be',
            'nl': 'nl-be',
            'zh-cn': 'zh', 'zh-hans': 'zh', 'zh-sg': 'zh',
            'zh-tw': 'zh', 'zh-hant': 'zh',
            'pt-br': 'pt', 'pt-pt': 'pt'
        }

    # -------------- Public --------------

    def convert_to_natural_speech(self, flight_response: str, detected_language: str = 'en') -> str:
        try:
            details = self._extract_flight_details_enhanced(flight_response)
            if not details:
                return self._clean_for_basic_speech(flight_response)

            lang = detected_language.lower()
            lang = self.lang_aliases.get(lang, lang)

            if lang in self.LANG_STRINGS:
                return self._generate_lang_speech(details, lang)
            # keep old specific ones if needed
            if lang in ['ur', 'hi', 'ar']:
                # already covered above but safe
                return self._generate_lang_speech(details, lang)
            # default english
            return self._generate_lang_speech(details, 'en')

        except Exception:
            return self._clean_for_basic_speech(flight_response)

    # -------------- Extraction --------------

    def _extract_flight_details_enhanced(self, response: str) -> Optional[Dict]:
        def grab(patterns: List[str]) -> Optional[str]:
            for p in patterns:
                m = re.search(p, response, flags=re.IGNORECASE)
                if m:
                    return m.group(1).strip()
            return None

        details: Dict[str, str] = {}

        # --- Detect round-trip sections if present ---
        lower = response.lower()
        out_idx = lower.find('outbound flight')
        ret_idx = lower.find('return flight')
        def section_text(start: int, end: Optional[int]) -> str:
            if start == -1:
                return ''
            if end is None or end == -1:
                return response[start:]
            return response[start:end]

        out_section = section_text(out_idx, ret_idx if ret_idx != -1 else None)
        ret_section = section_text(ret_idx, None)

        # Helper to parse a section (outbound/return)
        def parse_leg(section: str) -> Dict[str, str]:
            leg: Dict[str, str] = {}
            if not section:
                return leg
            def sgrab(patterns: List[str]) -> Optional[str]:
                for p in patterns:
                    m = re.search(p, section, flags=re.IGNORECASE)
                    if m:
                        return m.group(1).strip()
                return None
            dep_raw = sgrab([r'📅\s*Departure:\s*(.+?)(?:\n|$)', r'Departure:\s*(.+?)(?:\n|$)'])
            arr_raw = sgrab([r'🛬\s*Arrival:\s*(.+?)(?:\n|$)', r'Arrival:\s*(.+?)(?:\n|$)'])
            airline = sgrab([r'🏢\s*Airline:\s*(.+?)(?:\n|$)', r'Airline:\s*(.+?)(?:\n|$)'])
            flight_no = sgrab([r'✈️\s*Flight:\s*(.+?)(?:\n|$)', r'Flight:\s*(.+?)(?:\n|$)'])
            stops = sgrab([r'🔄\s*Stops:\s*(.+?)(?:\n|$)', r'Stops?:\s*(.+?)(?:\n|$)'])
            dur = sgrab([r'⏱️\s*Duration:\s*(.+?)(?:\n|$)', r'Duration:\s*(.+?)(?:\n|$)'])

            def parse_dt_airport(raw: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
                pats = [
                    r'(\w+\s+\d{1,2}(?:,\s*\d{4})?)\s*(?:at\s*)?(\d{1,2}:\d{2})\s*\((\w{3})\)',
                    r'(\d{1,2}\s+\w+\s*\d{4}?)\s*(\d{1,2}:\d{2})\s*\((\w{3})\)',
                    r'(\d{4}-\d{2}-\d{2})\s*(\d{1,2}:\d{2})\s*\((\w{3})\)'
                ]
                for p in pats:
                    m = re.search(p, raw)
                    if m:
                        return m.group(1), m.group(2), m.group(3)
                return None, None, None

            if dep_raw:
                d_date, d_time, d_code = parse_dt_airport(dep_raw)
                if d_date: leg['departure_date'] = d_date
                if d_time: leg['departure_time'] = d_time
                if d_code: leg['from_city'] = d_code
                leg['departure_info'] = dep_raw
            if arr_raw:
                a_date, a_time, a_code = parse_dt_airport(arr_raw)
                if a_date: leg['arrival_date'] = a_date
                if a_time: leg['arrival_time'] = a_time
                if a_code: leg['to_city'] = a_code
                leg['arrival_info'] = arr_raw
            if airline: leg['airline'] = airline
            if flight_no: leg['flight_number'] = flight_no
            if stops: leg['stops'] = stops
            if dur: leg['duration'] = dur
            return leg

        outbound_leg = parse_leg(out_section)
        return_leg = parse_leg(ret_section)

        if outbound_leg and return_leg:
            details['roundtrip'] = True
            # Prefix fields to avoid collision
            for k, v in outbound_leg.items():
                details[f'out_{k}'] = v
            for k, v in return_leg.items():
                details[f'ret_{k}'] = v
            # Also capture price and baggage from whole response
            m = re.search(r'(?:💰\s*)?Total Price:\s*(\w+)\s*([\d,\.]+)', response, flags=re.IGNORECASE)
            if not m:
                m = re.search(r'(?:💰\s*)?Price:\s*(\w+)\s*([\d,\.]+)', response, flags=re.IGNORECASE)
            if m:
                details['currency'] = m.group(1)
                details['price'] = m.group(2).replace(',', '')
            bag = grab([r'🧳\s*Baggage:\s*(.+?)(?:\n|$)', r'Baggage:\s*(.+?)(?:\n|$)'])
            if bag:
                details['baggage'] = bag
            total_trip = grab([r'⏰\s*Total Trip Duration:\s*(.+?)(?:\n|$)'])
            if total_trip:
                details['total_trip_duration'] = total_trip
            return details

        # Price
        m = re.search(r'(?:💰\s*)?Price:\s*(\w+)\s*([\d,\.]+)', response, flags=re.IGNORECASE)
        if not m:
            m = re.search(r'(?:Fare|Cost):\s*(\w+)\s*([\d,\.]+)', response, flags=re.IGNORECASE)
        if m:
            details['currency'] = m.group(1)
            details['price'] = m.group(2).replace(',', '')

        # Departure & arrival lines
        dep_raw = grab([r'🛫\s*Departure:\s*(.+?)(?:\n|$)',
                        r'Departure:\s*(.+?)(?:\n|$)',
                        r'Leave[s]?:\s*(.+?)(?:\n|$)'])
        arr_raw = grab([r'🛬\s*Arrival:\s*(.+?)(?:\n|$)',
                        r'Arrival:\s*(.+?)(?:\n|$)',
                        r'Arrive[s]?:\s*(.+?)(?:\n|$)'])

        details['departure_info'] = dep_raw or ''
        details['arrival_info'] = arr_raw or ''

        # Extract components
        def parse_dt_airport(raw: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
            patterns = [
                r'(\w+\s+\d{1,2}(?:,\s*\d{4})?)\s*(?:at\s*)?(\d{1,2}:\d{2})\s*\((\w{3})\)',
                r'(\d{1,2}\s+\w+\s*\d{4}?)\s*(\d{1,2}:\d{2})\s*\((\w{3})\)',
                r'(\d{4}-\d{2}-\d{2})\s*(\d{1,2}:\d{2})\s*\((\w{3})\)'
            ]
            for p in patterns:
                m = re.search(p, raw)
                if m:
                    return m.group(1), m.group(2), m.group(3)
            return None, None, None

        dep_date, dep_time, dep_code = parse_dt_airport(details['departure_info'])
        arr_date, arr_time, arr_code = parse_dt_airport(details['arrival_info'])

        if dep_date: details['departure_date'] = dep_date
        if dep_time: details['departure_time'] = dep_time
        if dep_code: details['from_city'] = dep_code

        if arr_date: details['arrival_date'] = arr_date
        if arr_time: details['arrival_time'] = arr_time
        if arr_code: details['to_city'] = arr_code

        # Airline, flight number, duration, stops, baggage
        details['airline'] = grab([r'🏢\s*Airline:\s*(.+?)(?:\n|$)',
                                   r'Airline:\s*(.+?)(?:\n|$)']) or ''
        details['flight_number'] = grab([r'✈️\s*Flight:\s*(.+?)(?:\n|$)',
                                         r'Flight:\s*(.+?)(?:\n|$)']) or ''
        details['stops'] = grab([r'🔄\s*Stops:\s*(.+?)(?:\n|$)',
                                 r'Stops?:\s*(.+?)(?:\n|$)']) or ''
        details['duration'] = grab([r'⏱️\s*Duration:\s*(.+?)(?:\n|$)',
                                    r'Duration:\s*(.+?)(?:\n|$)']) or ''
        details['baggage'] = grab([r'🧳\s*Baggage:\s*(.+?)(?:\n|$)',
                                   r'Baggage:\s*(.+?)(?:\n|$)']) or ''

        # Strip N/A
        for k, v in list(details.items()):
            if isinstance(v, str) and v.strip().upper() == 'N/A':
                details[k] = ''

        return details if any(details.values()) else None

    # -------------- Multilingual Speech Generator --------------

    def _generate_lang_speech(self, d: Dict, lang: str) -> str:
        L = self.LANG_STRINGS[lang]
        parts: List[str] = []

        parts.append(L['greeting'])

        if d.get('price'):
            parts.append(L['price'].format(price=d['price'],
                                           currency=self._get_currency_name(d.get('currency', ''))))

        # One-way airline line
        if d.get('airline') and not d.get('roundtrip'):
            parts.append(L['airline'].format(airline=self._get_airline_name(d['airline'])))

        if d.get('flight_number') and not d.get('roundtrip'):
            parts.append(L['flight_no'].format(flight=d['flight_number']))

        if d.get('roundtrip'):
            # Outbound block
            parts.append("Outbound:")
            out_dep = self._build_dep_arr_phrase_generic(
                L['leaves'], d.get('out_departure_date'), d.get('out_departure_time'),
                d.get('out_from_city'), d.get('out_departure_info', ''), L, dep=True
            )
            if out_dep:
                parts.append(out_dep)
            out_arr = self._build_dep_arr_phrase_generic(
                L['lands'], d.get('out_arrival_date'), d.get('out_arrival_time'),
                d.get('out_to_city'), d.get('out_arrival_info', ''), L, dep=False
            )
            if out_arr:
                parts.append(out_arr)
            if d.get('out_duration'):
                parts.append(L['total_time'].format(duration=self._clean_duration(d['out_duration'])))
            if d.get('out_stops'):
                st_low = d['out_stops'].lower()
                if "direct" in st_low or "nonstop" in st_low or "0" in st_low:
                    parts.append(L['direct'])
                else:
                    parts.append(L['has_stops'].format(stops=d['out_stops']))
            if d.get('out_airline'):
                parts.append(L['airline'].format(airline=self._get_airline_name(d['out_airline'])))
            if d.get('out_flight_number'):
                parts.append(L['flight_no'].format(flight=d['out_flight_number']))

            # Return block
            parts.append("Return:")
            ret_dep = self._build_dep_arr_phrase_generic(
                L['leaves'], d.get('ret_departure_date'), d.get('ret_departure_time'),
                d.get('ret_from_city'), d.get('ret_departure_info', ''), L, dep=True
            )
            if ret_dep:
                parts.append(ret_dep)
            ret_arr = self._build_dep_arr_phrase_generic(
                L['lands'], d.get('ret_arrival_date'), d.get('ret_arrival_time'),
                d.get('ret_to_city'), d.get('ret_arrival_info', ''), L, dep=False
            )
            if ret_arr:
                parts.append(ret_arr)
            if d.get('ret_duration'):
                parts.append(L['total_time'].format(duration=self._clean_duration(d['ret_duration'])))
            if d.get('ret_stops'):
                st_low = d['ret_stops'].lower()
                if "direct" in st_low or "nonstop" in st_low or "0" in st_low:
                    parts.append(L['direct'])
                else:
                    parts.append(L['has_stops'].format(stops=d['ret_stops']))
            if d.get('ret_airline'):
                parts.append(L['airline'].format(airline=self._get_airline_name(d['ret_airline'])))
            if d.get('ret_flight_number'):
                parts.append(L['flight_no'].format(flight=d['ret_flight_number']))

            # Total trip duration if provided – always include if present
            if d.get('total_trip_duration'):
                parts.append(L['total_time'].format(duration=self._clean_duration(d['total_trip_duration'])))
            # If not present, but both leg durations are present, include a concise combined reminder
            elif d.get('out_duration') and d.get('ret_duration'):
                parts.append("Both ways combined look good timing-wise.")
        else:
            # One-way formatting (existing)
            dep_phrase = self._build_dep_arr_phrase_generic(
                L['leaves'], d.get('departure_date'), d.get('departure_time'),
                d.get('from_city'), d.get('departure_info'), L, dep=True
            )
            if dep_phrase:
                parts.append(dep_phrase)
            arr_phrase = self._build_dep_arr_phrase_generic(
                L['lands'], d.get('arrival_date'), d.get('arrival_time'),
                d.get('to_city'), d.get('arrival_info'), L, dep=False
            )
            if arr_phrase:
                parts.append(arr_phrase)
            if d.get('duration'):
                parts.append(L['total_time'].format(duration=self._clean_duration(d['duration'])))
            if d.get('stops'):
                st_low = d['stops'].lower()
                if "direct" in st_low or "nonstop" in st_low or "0" in st_low:
                    parts.append(L['direct'])
                else:
                    parts.append(L['has_stops'].format(stops=d['stops']))

        # Baggage
        if d.get('baggage'):
            text = d['baggage']
            t = text.lower()
            if 'fee' in t or 'apply' in t:
                parts.append(L['baggage_fee'])
            elif 'included' in t or '1pc' in t or 'one piece' in t:
                parts.append(L['baggage_inc'])
            else:
                parts.append(L['baggage_generic'].format(text=text))

        parts.append(L['close'])

        return self._join_speech_parts(parts)

    # -------------- Helpers --------------

    def _build_dep_arr_phrase_generic(self, verb: str, date_str: Optional[str], time_str: Optional[str],
                                      code: Optional[str], raw_line: str,
                                      L: Dict[str, str], dep: bool) -> Optional[str]:
        if not raw_line and not date_str and not time_str and not code:
            return None

        city = self._get_city_name(code) if code else ''
        date_spoken = self._speak_date(date_str) if date_str else ''
        time_spoken = self._speak_time(time_str) if time_str else ''

        if not (date_spoken or time_spoken or city):
            return self._clean_time_info(raw_line)

        pieces = [verb]
        if date_spoken and L['on']:
            pieces.append(f"{L['on']} {date_spoken}")
        if time_spoken and L['at']:
            pieces.append(f"{L['at']} {time_spoken}")
        if city:
            pieces.append(f"{L['from']} {city}" if dep else f"{L['to']} {city}")

        return self._human_join(pieces, commas=True) + "."

    def _speak_date(self, date_raw: Optional[str]) -> str:
        if not date_raw:
            return ''
        date_raw = date_raw.strip().replace(',', '')
        m = re.match(r'([A-Za-z]{3,})\s+0?(\d{1,2})(?:\s+(\d{4}))?', date_raw)
        if m:
            month_name = self._full_month_name(m.group(1))
            day = int(m.group(2))
            year = m.group(3)
            return f"{day} {month_name}" + (f" {year}" if year else '')

        m = re.match(r'0?(\d{1,2})\s+([A-Za-z]{3,})(?:\s+(\d{4}))?', date_raw)
        if m:
            day = int(m.group(1))
            month_name = self._full_month_name(m.group(2))
            year = m.group(3)
            return f"{day} {month_name}" + (f" {year}" if year else '')

        m = re.match(r'(\d{4})-(\d{2})-(\d{2})', date_raw)
        if m:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            month_name = datetime(year, month, day).strftime("%B")
            return f"{day} {month_name} {year}"

        return re.sub(r'\b0([1-9])\b', r'\1', date_raw)

    def _speak_time(self, time_raw: Optional[str]) -> str:
        if not time_raw:
            return ''
        m = re.match(r'0?(\d{1,2}):(\d{2})', time_raw.strip())
        if not m:
            return time_raw
        hour = int(m.group(1))
        minute = m.group(2)
        if minute == "00":
            return f"{hour}:00"
        return f"{hour}:{minute}"

    def _clean_duration(self, dur: str) -> str:
        h_m = re.findall(r'(\d+)\s*h', dur, flags=re.IGNORECASE)
        m_m = re.findall(r'(\d+)\s*m', dur, flags=re.IGNORECASE)
        hours = f"{h_m[0]} hour{'s' if h_m and h_m[0] != '1' else ''}" if h_m else ''
        mins = f"{m_m[0]} minute{'s' if m_m and m_m[0] != '1' else ''}" if m_m else ''
        return self._human_join([hours, mins]) if hours or mins else dur

    def _get_city_name(self, code: str) -> str:
        return self.city_names.get(code, code)

    def _get_airline_name(self, airline: str) -> str:
        if len(airline) <= 3:
            return self.airline_names.get(airline, airline)
        return airline

    def _get_currency_name(self, currency: str) -> str:
        return self.currency_names.get(currency, currency)

    def _join_speech_parts(self, parts: List[str]) -> str:
        spoken = " ".join([p.strip() for p in parts if p.strip()])
        return re.sub(r'\s+', ' ', spoken).strip()

    def _clean_for_basic_speech(self, text: str) -> str:
        """Minimal fallback cleaner when we can't extract structured details.
        Removes emoji/markup and condenses whitespace so TTS sounds natural.
        """
        if not text:
            return ""
        # Remove most emojis and pictographs
        cleaned = re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\u2600-\u26FF]", "", text)
        # Remove common icons used in our messages
        cleaned = re.sub(r"[✈️🛫🛬💰🔄🧳⏱️📅🏢🎉✅❌⚠️]", "", cleaned)
        # Normalize punctuation spacing
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _clean_time_info(self, time_info: str) -> str:
        cleaned = re.sub(r'[🛫🛬]', '', time_info)
        cleaned = cleaned.replace('Terminal M', 'from Terminal M')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def _human_join(self, items: List[str], commas: bool = False) -> str:
        items = [i for i in items if i]
        if not items:
            return ''
        if len(items) == 1:
            return items[0]
        if commas:
            return ', '.join(items[:-1]) + ' and ' + items[-1]
        return ' '.join(items)

    def _full_month_name(self, token: str) -> str:
        try:
            idx = self.months_short[token.lower()]
            return datetime(2000, idx, 1).strftime("%B")
        except Exception:
            try:
                dt = datetime.strptime(token[:3], "%b")
                return dt.strftime("%B")
            except Exception:
                return token.capitalize()


# Global instance
flight_speech_formatter = FlightSpeechFormatter()
