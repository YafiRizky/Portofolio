import json
import re
import random
import logging
from typing import Dict, List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass

# Import pustaka yang diperlukan untuk model AI
try:
    from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
except ImportError:
    print("Peringatan: Pustaka 'transformers' tidak ditemukan. Bot akan berjalan dalam mode rule-based saja.")
    BlenderbotTokenizer, BlenderbotForConditionalGeneration = None, None

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FlowDefinition:
    steps: List[Callable]
    description: str = "A guided conversation flow."

class ConversationContext:
    def __init__(self):
        self.active_flow: Optional[str] = None
        self.flow_step: int = 0
        self.data: Dict = {}
        self.conversation_history: List[str] = []

    def start_flow(self, flow_name: str) -> None:
        self.active_flow = flow_name
        self.flow_step = 0
        self.data = {}
        logger.info(f"New flow started: '{flow_name}'")

    def advance_flow(self) -> None:
        if self.active_flow:
            self.flow_step += 1
            logger.info(f"Flow '{self.active_flow}' advanced to step {self.flow_step}")

    def add_to_history(self, message: str) -> None:
        self.conversation_history.append(message)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def get_history_string(self) -> str:
        """Menggabungkan riwayat percakapan menjadi satu string."""
        return "\n".join(self.conversation_history)

    def reset(self) -> None:
        logger.info("Conversation context completely reset.")
        self.__init__()

class MentalHealthChatbot:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / 'data'
        
        self.knowledge_base = self._load_data('knowledge_base.json')
        self.qa_pairs = self._load_data('qa_pairs.json')
        self.emergency_data = self._load_data('emergency.json')
        
        self.context = ConversationContext()
        self._initialize_topic_mapping()
        self._initialize_guided_flows()
        
        self.emergency_keywords = self.emergency_data.get('emergency_keywords', [])

        # [BARU] Inisialisasi dan pemuatan model AI
        self.ml_tokenizer = None
        self.ml_model = None
        self._load_ml_model()

    def _load_data(self, filename: str) -> Dict:
        # (Fungsi ini tetap sama)
        file_path = self.data_dir / filename
        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}.")
            return {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {}

    def _load_ml_model(self):
        """[FUNGSI BARU] Memuat model AI BlenderBot dari Hugging Face."""
        if BlenderbotTokenizer is None:
            logger.warning("Tidak bisa memuat model AI karena pustaka 'transformers' tidak ada.")
            return

        try:
            logger.info("Mencoba memuat model AI BlenderBot (mungkin butuh waktu saat pertama kali)...")
            model_name = "facebook/blenderbot_small-90M"
            self.ml_tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
            self.ml_model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
            logger.info("Model AI BlenderBot berhasil dimuat.")
        except Exception as e:
            logger.error(f"Gagal memuat model AI: {e}")
            logger.warning("Chatbot akan berjalan dalam mode rule-based saja.")
            self.ml_tokenizer = None
            self.ml_model = None
            
    # ... (Semua fungsi _initialize dan _..._step tetap sama persis seperti sebelumnya) ...
    def _initialize_topic_mapping(self):
        self.topic_keywords = {
            "stres": ["stres", "stress", "tertekan", "beban", "tekanan", "kewalahan", "overwhelmed"],
            "kecemasan": ["cemas", "khawatir", "gelisah", "anxiety", "panik", "takut", "gugup"],
            "depresi": ["sedih", "hampa", "kosong", "putus asa", "tidak bersemangat", "murung", "depresi"],
            "marah": ["marah", "kesal", "jengkel", "emosi", "geram", "frustrasi", "benci"],
            "overthinking": ["overthinking", "pikiran berputar", "tidak bisa berhenti mikir", "kepikiran terus"],
            "insomnia": ["tidak bisa tidur", "susah tidur", "insomnia", "begadang", "sulit tidur"],
            "perpisahan": ["putus", "pisah", "patah hati", "kehilangan", "ditinggal", "berakhir"],
            "kesepian": ["kesepian", "sendiri", "lonely", "terisolasi", "tidak ada teman"],
            "masalah_keluarga": ["keluarga", "orang tua", "adik", "kakak", "konflik keluarga"],
            "media_sosial": ["media sosial", "instagram", "facebook", "tiktok", "sosmed", "comparing"],
            "quarter_life_crisis": ["bingung hidup", "tidak tahu arah", "quarter life", "masa depan"],
            "motivasi": ["tidak semangat", "malas", "tidak ada motivasi", "kehilangan semangat"],
            "self_care": ["merawat diri", "self care", "lelah", "butuh istirahat", "me time"],
            "trauma": ["trauma", "masa lalu", "terluka", "sakit hati mendalam", "ptsd"],
            "burnout": ["burnout", "kelelahan kerja", "jenuh kerja", "capek bekerja"],
            "bpd": ["bpd", "borderline", "borderline personality disorder", "gangguan kepribadian"]
        }

    def _find_relevant_topic(self, text: str) -> Optional[str]:
        if not text: return None
        text_lower = text.lower().strip()
        topic_scores = {
            topic: sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower))
            for topic, keywords in self.topic_keywords.items()
        }
        scored_topics = {topic: score for topic, score in topic_scores.items() if score > 0}
        return max(scored_topics, key=scored_topics.get) if scored_topics else None

    def _initialize_guided_flows(self) -> None:
        self._guided_flows = {
            "stres": FlowDefinition([self._stres_step0, self._stres_step1]),
            "kecemasan": FlowDefinition([self._kecemasan_step0, self._kecemasan_step1]),
            "depresi": FlowDefinition([self._depresi_step0, self._depresi_step1]),
            "marah": FlowDefinition([self._marah_step0, self._marah_step1]),
            "overthinking": FlowDefinition([self._overthinking_step0, self._overthinking_step1, self._overthinking_step2]),
            "insomnia": FlowDefinition([self._insomnia_step0, self._insomnia_step1]),
            "perpisahan": FlowDefinition([self._perpisahan_step0, self._perpisahan_step1]),
            "kesepian": FlowDefinition([self._kesepian_step0, self._kesepian_step1]),
            "masalah_keluarga": FlowDefinition([self._masalah_keluarga_step0, self._masalah_keluarga_step1]),
            "media_sosial": FlowDefinition([self._media_sosial_step0, self._media_sosial_step1]),
            "quarter_life_crisis": FlowDefinition([self._qlc_step0, self._qlc_step1]),
            "motivasi": FlowDefinition([self._motivasi_step0, self._motivasi_step1]),
            "self_care": FlowDefinition([self._self_care_step0, self._self_care_step1]),
            "bpd": FlowDefinition([self._bpd_step0]),
            "trauma": FlowDefinition([self._trauma_step0, self._trauma_step1]),
            "burnout": FlowDefinition([self._burnout_step0, self._burnout_step1]),
        }

    def _get_contextual_response(self, suggested_topic: str) -> str:
        topic_display = suggested_topic.replace('_', ' ').title()
        return (
            f"Dari yang kamu ceritakan, sepertinya ini berkaitan dengan topik "
            f"<strong>'{topic_display}'</strong>. Apakah kamu mau kita bahas topik ini "
            f"lebih dalam? Kamu bisa klik topik tersebut atau balas dengan '{suggested_topic}' untuk memulai."
        )

    def _stres_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Tentu, mari kita bahas tentang stres. Aku dengar kamu sedang merasa tertekan. Perasaan itu valid.<br><br>Boleh ceritakan sedikit, apa hal spesifik yang paling membuatmu merasa stres akhir-akhir ini?"""

    def _stres_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Terima kasih sudah berbagi. Menghadapi '{user_input}' memang tidak mudah. Mengakui sumber stres adalah langkah pertama yang hebat.<br><br>Ingat, fokus pada satu hal kecil yang bisa kamu kontrol saat ini. Kamu tidak harus menyelesaikan semuanya sekaligus."""

    def _kecemasan_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Mari kita bicara tentang kecemasan. Rasa khawatir dan gelisah itu sangat menguras energi.<br><br>Saat rasa cemas itu datang, apa yang biasanya kamu rasakan di tubuhmu? (Contoh: jantung berdebar, napas pendek, tangan dingin)"""

    def _kecemasan_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Terima kasih telah menjelaskannya. Mengenali respons tubuh adalah langkah penting. Saat itu terjadi lagi, coba satu hal ini: Tarik napas perlahan selama 4 detik, tahan 4 detik, lalu hembuskan perlahan selama 6 detik. Lakukan beberapa kali. Ini dapat membantu menenangkan sistem sarafmu."""

    def _depresi_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Aku di sini bersamamu untuk membahas perasaan sedih dan hampa. Kamu tidak sendirian.<br><br>Selain merasa sedih, adakah aktivitas yang dulu kamu nikmati tapi sekarang terasa tidak menarik lagi?"""

    def _depresi_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Kehilangan minat atau 'anhedonia' adalah gejala yang sangat umum. Terima kasih sudah jujur. Tidak apa-apa jika saat ini terasa berat. Bisakah kita pikirkan satu hal SANGAT kecil yang mungkin bisa kamu coba lakukan besok? (Contoh: duduk di luar selama 5 menit, atau mendengarkan satu lagu favorit)."""

    def _marah_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Rasa marah dan frustrasi adalah emosi yang kuat. Tidak apa-apa merasakannya.<br><br>Jika kamu nyaman, coba gambarkan: kemarahan ini terasa seperti apa? Apakah seperti api yang membakar, atau tekanan yang akan meledak?"""

    def _marah_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Deskripsi yang kuat. Terkadang, di balik kemarahan ada perasaan lain seperti sakit hati atau ketidakadilan. Mengakui emosi ini adalah langkah awal untuk mengelolanya secara sehat, bukan menekannya."""
        
    def _overthinking_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Tentu, mari kita coba jinakkan pikiran yang berputar-putar itu. Overthinking sangat melelahkan.<br><br>Langkah pertama, coba tuliskan satu pikiran negatif spesifik yang paling sering muncul di kepalamu."""

    def _overthinking_step1(self, user_input: str) -> str:
        self.context.data['negative_thought'] = user_input
        self.context.advance_flow()
        return f"""Oke, pikiranmu adalah: '<i>{user_input}</i>'.<br><br>Sekarang, mari kita uji. Apa satu bukti kuat yang mendukung pikiran ini? Dan apa satu bukti kuat yang membantahnya?"""

    def _overthinking_step2(self, user_input: str) -> str:
        self.context.reset()
        return f"""Bagus sekali. Kamu sudah mulai melihatnya dari dua sisi. Ini adalah keterampilan yang hebat. Latihan ini membantu otak kita untuk tidak langsung percaya pada pikiran negatif pertama yang muncul."""

    def _insomnia_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Sulit tidur memang sangat mengganggu. Mari kita lihat.<br><br>Apa yang biasanya ada di pikiranmu atau kamu lakukan satu jam sebelum mencoba untuk tidur?"""

    def _insomnia_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Terima kasih. Seringkali, apa yang kita lakukan sebelum tidur (screen time, memikirkan kerjaan) sangat berpengaruh. Menciptakan 'zona tenang' satu jam sebelum tidur tanpa gadget bisa membuat perbedaan besar."""

    def _perpisahan_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Patah hati karena perpisahan itu nyata dan menyakitkan. Perasaanmu sangat valid.<br><br>Siapa yang kamu rindukan saat ini? Kamu tidak perlu menyebut nama, cukup perannya dalam hidupmu (misalnya: 'sahabat baik' atau 'pasangan')."""

    def _perpisahan_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Kehilangan seorang {user_input} meninggalkan ruang kosong. Izinkan dirimu untuk berduka. Tidak ada batas waktu untuk pulih. Merawat dirimu sendiri saat ini adalah prioritas utama."""

    def _kesepian_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Perasaan kesepian itu berat, seolah tak terlihat. Aku melihatmu dan aku di sini mendengarkan.<br><br>Jika kamu bisa memilih, koneksi seperti apa yang paling kamu dambakan saat ini?"""

    def _kesepian_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Mendambakan '{user_input}' itu sangat manusiawi. Langkah pertama untuk keluar dari kesepian adalah dengan jujur pada keinginan itu. Terima kasih sudah terbuka."""
    
    def _masalah_keluarga_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Konflik keluarga bisa sangat menguras energi dan menyakitkan karena terjadi di tempat yang seharusnya aman.<br><br>Tanpa perlu detail, perasaan apa yang paling dominan saat kamu memikirkan masalah ini? (Contoh: marah, sedih, lelah, kecewa)"""

    def _masalah_keluarga_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Merasa {user_input} adalah respons yang sangat wajar dalam situasi seperti itu. Ingat, kamu berhak memiliki batasan untuk melindungi kedamaian mentalmu, bahkan dari keluarga."""

    def _media_sosial_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Lelah dengan media sosial itu sangat umum sekarang. Terkadang apa yang kita lihat di sana membuat kita merasa kurang.<br><br>Aplikasi atau konten seperti apa yang paling sering membuatmu merasa buruk tentang dirimu sendiri?"""

    def _media_sosial_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Itu wawasan yang bagus. Menyadari pemicunya adalah langkah besar. Mungkin kamu bisa mencoba fitur 'mute' atau 'unfollow' akun-akun tersebut? Kamu berhak menciptakan linimasa yang mendukung, bukan yang menjatuhkan."""

    def _qlc_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Ah, 'quarter-life crisis'. Merasa bingung, tersesat, dan membandingkan diri dengan orang lain. Sangat umum dan sangat berat.<br><br>Dari semua aspek (karir, hubungan, tujuan hidup), mana yang terasa paling tidak pasti saat ini?"""

    def _qlc_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Fokus pada ketidakpastian di '{user_input}' itu bisa membuat kewalahan. Ingat, tidak ada orang yang punya semua jawaban. Tidak apa-apa untuk tidak tahu. Langkahmu saat ini adalah bertahan dan terus mencoba hal-hal kecil."""

    def _motivasi_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Kehilangan motivasi itu seperti mobil kehabisan bensin. Bukan mobilnya yang rusak, hanya butuh bahan bakar.<br><br>Apa satu hal yang jika berhasil kamu lakukan, akan membuatmu merasa sedikit lebih baik, sekecil apapun itu?"""

    def _motivasi_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""'{user_input}' terdengar seperti tujuan yang bagus. Coba pecah menjadi langkah yang SANGAT KECIL. Apa langkah paling pertama yang bisa kamu ambil untuk itu?"""

    def _self_care_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Merawat diri atau 'self-care' bukan kemewahan, tapi kebutuhan. Ini tentang mengisi kembali energimu.<br><br>Apa aktivitas self-care favoritmu, atau apa yang ingin kamu coba lakukan untuk dirimu sendiri?"""

    def _self_care_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Melakukan '{user_input}' terdengar sangat menenangkan. Aku harap kamu bisa meluangkan waktu untuk itu. Kamu pantas mendapatkannya."""

    def _trauma_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Membicarakan trauma itu berat, dan aku di sini untuk mendengarkan dengan hati-hati. Keamananmu adalah yang utama.<br><br>Saat ini, apa yang kamu butuhkan untuk merasa sedikit lebih aman atau tenang?"""

    def _trauma_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Terima kasih. Fokus pada kebutuhanmu saat ini ('ingin merasa aman') adalah hal yang tepat. Jika kamu merasa kewalahan, teknik grounding (menyebutkan 5 benda yang kamu lihat) bisa membantu menarikmu kembali ke saat ini."""
    
    def _burnout_step0(self, _: Optional[str] = None) -> str:
        self.context.advance_flow()
        return f"""Kelelahan kerja atau 'burnout' lebih dari sekadar lelah biasa. Ini adalah kelelahan emosional, fisik, dan mental yang mendalam.<br><br>Gejala mana yang paling kamu rasakan: kelelahan total, sinisme/sikap negatif terhadap pekerjaan, atau merasa tidak kompeten?"""

    def _burnout_step1(self, user_input: str) -> str:
        self.context.reset()
        return f"""Merasakan '{user_input}' adalah tanda jelas dari burnout. Ini bukan salahmu, ini adalah respons terhadap stres kronis. Istirahat yang sesungguhnya—bukan hanya libur tapi benar-benar lepas dari pekerjaan—sangatlah penting."""

    def _bpd_step0(self, _: Optional[str] = None) -> str:
        self.context.reset()
        info = self.knowledge_base.get('bpd', {})
        definisi = info.get('definition', 'Informasi tidak ditemukan.')
        return f"""Tentu, ini adalah informasi umum mengenai <strong>Borderline Personality Disorder (BPD)</strong>.<br><br><strong>Definisi:</strong> {definisi}<br><br>Ini adalah kondisi kompleks yang memerlukan diagnosis dan perawatan dari profesional. Jika kamu merasa ini relevan, berbicara dengan psikolog adalah langkah terbaik."""

    def generate_response(self, input_text: str) -> str:
        if not input_text or not input_text.strip():
            return "Aku di sini mendengarkan. Apa yang ingin kamu ceritakan?"
        
        input_text = input_text.strip()
        self.context.add_to_history(f"User: {input_text}")

        # 1. Emergency Check
        if (emergency_response := self._check_emergency(input_text)):
            return emergency_response

        # 2. Continue Active Flow
        if self.context.active_flow:
            return self._handle_active_flow(input_text)

        # 3. Start New Flow if input is an exact topic key
        if (response := self._handle_start_flow(input_text)):
            return response

        # 4. Handle General Q&A
        if (qa_response := self._get_qa_response(input_text)):
            return qa_response

        # 5. Suggest Topic based on keywords
        if (suggested_topic := self._find_relevant_topic(input_text)):
            return self._get_contextual_response(suggested_topic)

        # [PERUBAHAN UTAMA] 6. Gunakan Model AI sebagai Fallback Cerdas
        if (ml_response := self._generate_ml_response(input_text)):
            return ml_response

        # 7. Fallback terakhir jika semua gagal (termasuk model AI)
        return self._get_smart_fallback_response()
    
    def _handle_active_flow(self, user_input: str) -> Optional[str]:
        """Manages the logic for a continuing conversation."""
        try:
            flow_name = self.context.active_flow
            step_index = self.context.flow_step
            flow_def = self._guided_flows[flow_name]
            if step_index < len(flow_def.steps):
                step_function = flow_def.steps[step_index]
                response = step_function(user_input)
                if not self.context.active_flow:
                    response += "<br><br>Terima kasih sudah berbagi. Apakah ada topik lain?"
                return response
        except Exception as e:
            logger.error(f"Error during flow '{self.context.active_flow}': {e}")
            self.context.reset()
            return "Maaf, ada sedikit kendala. Mari kita mulai dari awal."
        return None

    def _handle_start_flow(self, user_input: str) -> Optional[str]:
        """Starts a new flow if the input is a recognized topic key."""
        input_key = user_input.lower().strip()
        if input_key in self._guided_flows:
            try:
                self.context.start_flow(input_key)
                step_function = self._guided_flows[input_key].steps[0]
                return step_function()
            except Exception as e:
                logger.error(f"Error starting flow '{input_key}': {e}")
                self.context.reset()
                return "Maaf, terjadi kesalahan saat memulai topik itu."
        return None
        
    def _generate_ml_response(self, text: str) -> Optional[str]:
        """[FUNGSI BARU] Menghasilkan respons menggunakan model AI jika tersedia."""
        if not self.ml_model or not self.ml_tokenizer:
            return None # Model tidak dimuat, jadi lewati langkah ini
        
        try:
            logger.info("Menggunakan model AI untuk menghasilkan respons...")
            # Menggunakan riwayat percakapan untuk konteks yang lebih baik
            conversation_history = self.context.get_history_string()
            
            inputs = self.ml_tokenizer(conversation_history, return_tensors="pt")
            reply_ids = self.ml_model.generate(**inputs)
            response = self.ml_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
            
            # Tambahkan respons bot ke riwayat
            self.context.add_to_history(f"Bot: {response}")
            return response
        except Exception as e:
            logger.error(f"Error saat menghasilkan respons dari model AI: {e}")
            return None

    def _check_emergency(self, text: str) -> Optional[str]:
        if not text or not self.emergency_keywords: return None
        text_lower = text.lower()
        if any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in self.emergency_keywords):
            responses = self.emergency_data.get('emergency_responses', {})
            return responses.get('bunuh_diri', "Keselamatanmu adalah prioritas. Segera hubungi bantuan darurat.")
        return None

    def _get_qa_response(self, text: str) -> Optional[str]:
        if not text or not self.qa_pairs: return None
        text_lower = text.lower()
        for data in self.qa_pairs.values():
            if any(re.search(r'\b' + re.escape(p) + r'\b', text_lower) for p in data.get('patterns', [])):
                return random.choice(data.get('responses', ["Maaf, aku tidak yakin."]))
        return None

    def _get_smart_fallback_response(self) -> str:
        return random.choice([
            "Aku ingin sekali membantu, tapi aku kurang mengerti. Bisakah kamu pilih topik dari daftar yang ada?",
            "Terima kasih sudah berbagi. Aku masih belajar. Mungkin kita bisa mulai dengan memilih topik yang paling sesuai?",
        ])

    def get_hotlines(self) -> List[Dict]:
        """Mengambil daftar hotline dari data untuk ditampilkan di UI."""
        return self.emergency_data.get('hotlines', [])