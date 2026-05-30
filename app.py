import json
import logging
import os
import wikipedia
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, LangDetectException
from textblob import TextBlob
from deep_translator import GoogleTranslator
import pyttsx3
import threading

# ==========================================
# CONFIGURAÇÃO DE LOGGING
# ==========================================
LOG_FILE = "chatbot.log"

# Configura logging
def setup_logging():
    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

# ==========================================
# CARREGAMENTO DINÂMICO DOS ARQUIVOS JSON
# ==========================================
knowledge_base = []
fact_categories = {}

# Carrega JSONs
def load_knowledge_base():
    global knowledge_base, fact_categories
    knowledge_base = []
    fact_categories = {}
    
    json_files = ["orkut_cultura.json", "orkut_historia.json", "orkut_tecnico.json"]
    for file_name in json_files:
        if os.path.exists(file_name):
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    category = data.get("categoria", "geral")
                    dados = data.get("dados", [])
                    for item in dados:
                        knowledge_base.append(item)
                        fact_categories[item] = category
                logger.info("JSON carregado: %s (%d fatos, categoria: %s)", file_name, len(dados), category)
            except Exception as e:
                logger.error("Erro ao carregar %s: %s", file_name, e, exc_info=True)
        else:
            logger.warning("Arquivo JSON não encontrado: %s", file_name)

    # Fallback se nenhum arquivo for carregado
    if not knowledge_base:
        logger.warning("Nenhum JSON carregado — usando base de conhecimento padrão (fallback)")
        knowledge_base = [
            "Orkut foi uma rede social criada em 2004 por Orkut Büyükkökten.",
            "O Orkut era muito popular no Brasil e na Índia.",
            "O Orkut tinha comunidades onde usuários podiam discutir interesses em comum.",
            "O Orkut foi descontinuado pelo Google em 2014.",
            "O Orkut permitia depoimentos públicos que os amigos podiam escrever no seu perfil.",
            "O Orkut tinha scraps, depoimentos e comunidades."
        ]
        for item in knowledge_base:
            fact_categories[item] = "geral"

# Inicializa a base
load_knowledge_base()
logger.info("Base de conhecimento carregada com %d fatos.", len(knowledge_base))

welcome_inputs = ["hi", "hello", "hey", "oi", "olá", "bom dia", "boa tarde", "boa noite"]
welcome_outputs = ["Olá! Seja bem-vindo ao Orkut!", "Oi! Como posso te ajudar hoje?", "Hey! Tudo tranquilo por aí?", "Olá! Pode perguntar sobre o Orkut :)"]

TYPO_FIXES = {"oorkut": "orkut", "orkutt": "orkut", "orkuttt": "orkut", "orcut": "orkut"}
STOP_WORDS_PT = [
    "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "e", "que", "foi", "ser", "eh", "é", "the", "is", "was", "por", "para",
    "com", "sem", "ao", "aos", "à", "às", "se", "sua", "seu", "suas", "seus", "meu", "minha",
    "aconteceu", "acontecer", "happened", "what", "with", "the",
]

TOPIC_KEYWORDS = {
    "jogo": ["jogo", "jogos", "game", "games", "aplicativo", "aplicativos", "app", "apps", "opensocial"],
    "scrap": ["scrap", "scraps", "recado", "recados", "scrapbook", "mural"],
    "comunidade": ["comunidade", "comunidades", "forum", "fórum"],
    "foto": ["foto", "fotos", "fotografia", "album", "álbum"],
    "amigo": ["amigo", "amigos", "amiga", "amizade", "depoimento", "depoimentos", "testemunho"],
    "encerramento": ["encerramento", "encerrar", "descontinuado", "descontinuar", "fim", "fechou", "fechamento", "shutdown"],
    "criacao": ["criado", "criou", "criação", "criar", "lançado", "lançamento", "fundado", "surgiu"],
}

# Detecta saudações
def welcome_message(text):
    words = re.sub(r'[^a-zA-Záéíóúãõâêôç ]', '', text.lower()).split()
    for word in welcome_inputs:
        if word in words:
            return random.choice(welcome_outputs)
    text_clean = re.sub(r'[^a-zA-Záéíóúãõâêôç ]', '', text.lower())
    for phrase in ["bom dia", "boa tarde", "boa noite"]:
        if phrase in text_clean:
            return random.choice(welcome_outputs)
    return None

stem_mapping = {
    "criou": "criar", "criaram": "criar", "criando": "criar", "criou-se": "criar",
    "criado": "criar", "criada": "criar",
    "comunidade": "comunidade", "comunidades": "comunidade",
    "amigo": "amigo", "amigos": "amigo", "amiga": "amigo", "amigas": "amigo", 
    "amizade": "amigo", "amizades": "amigo",
    "scrap": "scrap", "scraps": "scrap", "scrapbook": "scrap",
    "foto": "foto", "fotos": "foto", "fotografia": "foto", "fotografias": "foto",
    "popular": "popular", "popularização": "popular", "popularizou": "popular", 
    "popularizado": "popular", "popularidade": "popular"
}

# Corrige typos
def normalize_input(text):
    t = text.lower()
    for typo, fix in TYPO_FIXES.items():
        t = t.replace(typo, fix)
    return t

# Extrai tópicos
def extract_question_topics(text):
    t = normalize_input(text)
    return [topic for topic, keywords in TOPIC_KEYWORDS.items()
            if any(re.search(rf"\b{re.escape(kw)}\b", t) for kw in keywords)]

# Valida tópico
def fact_matches_topics(fact, topics):
    if not topics:
        return True
    t = normalize_input(fact)
    return any(
        re.search(rf"\b{re.escape(kw)}\b", t)
        for topic in topics for kw in TOPIC_KEYWORDS[topic]
    )

# Detecta intenção
def detect_intent(text):
    t = normalize_input(text)
    if re.search(r"\bquando\b", t):
        return "temporal"
    if re.search(r"\bquem\b", t):
        return "pessoa"
    if re.search(r"\bonde\b", t):
        return "local"
    if re.search(r"\b(como|por que|porque)\b", t):
        return "explicacao"
    return "geral"

# Reforça relevância
def intent_score_boost(fact, intent, user_text=""):
    bonus = 0.0
    if intent == "temporal":
        if re.search(r"\b(19|20)\d{2}\b", fact):
            bonus += 0.18
        if re.search(r"\b(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b", fact, re.I):
            bonus += 0.10
        if re.search(r"\b(criado|lançado|fundado|inaugurado|descontinuado|encerramento|encerrou)\b", fact, re.I):
            bonus += 0.12
        topic = temporal_topic(user_text)
        if topic != "geral" and temporal_topic(fact) == topic:
            bonus += 0.15
    elif intent == "pessoa":
        if re.search(r"\b(orkut büyükkökten|büyükkökten|criador|engenheiro|fundador|google)\b", fact, re.I):
            bonus += 0.15
    elif intent == "local":
        if re.search(r"\b(brasil|índia|estados unidos|turquia|país|países)\b", fact, re.I):
            bonus += 0.15
        if re.search(r"\b(onde|popular|país|países)\b", user_text, re.I):
            if re.search(r"\b(brasil|índia|estados unidos)\b", fact, re.I) and re.search(
                r"\b(segundo|primeiro|país|popular|ativo|atrás)\b", fact, re.I
            ):
                bonus += 0.25
    elif intent == "explicacao":
        if re.search(r"\b(porque|pois|devido|objetivo|motivo|razão)\b", fact, re.I):
            bonus += 0.10
    return bonus

# Classifica período
def temporal_topic(text):
    t = normalize_input(text)
    if re.search(r"\b(criado|criação|criou|lançado|fundado|surgiu|nasceu|inaugurado)\b", t):
        return "origem"
    if re.search(r"\b(descontinuado|encerr|encerrou|acabou|fim|morreu)\b", t):
        return "fim"
    return "geral"

# Reduz palavras
def simple_portuguese_stemmer(word):
    word = word.lower()
    if word in stem_mapping:
        return stem_mapping[word]
    if len(word) > 4:
        if word.endswith("s"):
            word = word[:-1]
    return word

# Limpa texto
def preprocess(sentence):
    sentence = normalize_input(sentence)
    sentence = re.sub(r'[^a-zA-Záéíóúãõâêôç ]', '', sentence)
    tokens = sentence.split()
    stemmed = [simple_portuguese_stemmer(t) for t in tokens]
    return " ".join(stemmed)

# Formata frase
def format_sentence(sentence):
    if not sentence:
        return ""
    words = sentence.split()
    if not words:
        return sentence
    first_word = words[0]
    if first_word in ["Orkut", "Brasil", "Índia", "Google", "HTML", "Java", "OpenSocial", "Takeout", "MySpace", "Facebook"]:
        return sentence
    return sentence[0].lower() + sentence[1:]

# Busca resposta
def get_answer(user_text, threshold=0.22):
    if not knowledge_base:
        return None

    intent = detect_intent(user_text)
    question_topics = extract_question_topics(user_text)
    cleaned_base = [preprocess(s) for s in knowledge_base]
    user_text_clean = preprocess(user_text)
    corpus = cleaned_base + [user_text_clean]

    try:
        tfidf = TfidfVectorizer(stop_words=STOP_WORDS_PT)
        matrix = tfidf.fit_transform(corpus)
    except Exception as e:
        logger.error("Erro NLP na vetorização TF-IDF: %s", e, exc_info=True)
        return None

    similarity = cosine_similarity(matrix[-1], matrix)[0]
    scores = list(similarity[:-1])

    for i, fact in enumerate(knowledge_base):
        scores[i] += intent_score_boost(fact, intent, user_text)
        if "orkut" in user_text_clean and "orkut" in fact.lower():
            scores[i] += 0.06

    for i, fact in enumerate(knowledge_base):
        scores[i] += intent_score_boost(fact, intent, user_text)
        if "orkut" in user_text_clean and "orkut" in fact.lower():
            scores[i] += 0.06
        if question_topics:
            if fact_matches_topics(fact, question_topics):
                scores[i] += 0.30
            else:
                scores[i] -= 0.40

    if question_topics:
        topic_indices = [i for i, fact in enumerate(knowledge_base) if fact_matches_topics(fact, question_topics)]
        if topic_indices:
            best_idx = max(topic_indices, key=lambda i: scores[i])
        else:
            logger.debug("Nenhum fato sobre tópicos %s", question_topics)
            return None
    else:
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_idx]

    if best_score < threshold:
        logger.debug("Nenhum match acima do threshold (%.2f). Melhor score: %.3f", threshold, best_score)
        return None

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if question_topics:
        ranked = [i for i in ranked if fact_matches_topics(knowledge_base[i], question_topics)]
    top_idx = ranked[0]
    responses = [knowledge_base[top_idx]]

    logger.debug(
        "Intent=%s | topics=%s | score=%.3f | fato=%s",
        intent, question_topics or "geral", best_score, knowledge_base[top_idx][:70],
    )

    for idx in ranked[1:3]:
        if scores[idx] < threshold:
            break
        if intent == "temporal" and temporal_topic(user_text) != "geral":
            if temporal_topic(knowledge_base[idx]) != temporal_topic(user_text):
                continue
        if scores[idx] / scores[top_idx] >= 0.88:
            responses.append(knowledge_base[idx])

    if len(responses) == 1:
        return responses[0]
    if len(responses) == 2:
        return f"{responses[0]} Além disso, {format_sentence(responses[1])}"
    return f"{responses[0]} Além disso, {format_sentence(responses[1])} Também vale ressaltar que {format_sentence(responses[2])}"

# Consulta Wikipedia
def buscar_no_wikipedia(query):
    query = normalize_input(query)
    try:
        wikipedia.set_lang("pt")
        results = wikipedia.search(query)
        if results:
            for result in results[:3]:
                try:
                    summary = wikipedia.summary(result, sentences=2)
                    logger.info("Wikipedia — resultado encontrado para '%s': %s", query, result)
                    return f"Não encontrei isso na base local do Orkut, mas pesquisei no Wikipédia e descobri o seguinte:\n\n\"{summary}\""
                except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
                    logger.debug("Wikipedia — resultado descartado '%s': %s", result, e)
                    continue
            logger.warning("Wikipedia — nenhum resultado válido para: %s", query)
        else:
            logger.info("Wikipedia — nenhum resultado encontrado para: %s", query)
    except Exception as e:
        logger.error("Erro ao buscar no Wikipedia para '%s': %s", query, e, exc_info=True)
    return None

# ==========================================
# ANÁLISE DE SENTIMENTO E IDIOMA
# ==========================================
# Verifica idioma
def check_language(text):
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "pt"

# Analisa sentimento
def get_sentiment_intervention(text):
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        blob = TextBlob(translated)
        polarity = blob.sentiment.polarity
        
        if polarity <= -0.3:
            return "Notei que você parece um pouco chateado. O Orkut era justamente um lugar para relaxar e fazer amigos! Mas, sobre o que você perguntou: "
    except Exception as e:
        logger.error("Erro NLP na análise de sentimento: %s", e, exc_info=True)
    return ""

PORTUGUESE_MARKERS = re.compile(
    r"\b(o|a|os|as|que|quem|quando|onde|como|por|para|do|da|em|foi|era|eram|"
    r"comunidade|depoimento|perfil|amigo|amigos)\b",
    re.I,
)
ENGLISH_MARKERS = re.compile(
    r"\b(when|what|who|where|how|why|was|were|is|are|the|you|create|created|hello|hey)\b",
    re.I,
)

# Traduz entrada
def translate_to_portuguese(text):
    lang = check_language(text)
    if lang == "pt":
        return text, "pt"
    if PORTUGUESE_MARKERS.search(text) and not ENGLISH_MARKERS.search(text):
        return text, "pt"
    try:
        translated = GoogleTranslator(source="auto", target="pt").translate(text)
        logger.info("Tradução %s→pt: '%s' → '%s'", lang, text, translated)
        return translated, lang
    except Exception as e:
        logger.error("Erro tradução: %s", e, exc_info=True)
        return text, lang

# Orquestra resposta
def chatbot_response(user_text):
    try:
        query_pt, original_lang = translate_to_portuguese(user_text)

        rule = welcome_message(user_text)
        if rule:
            return rule

        intervention = get_sentiment_intervention(query_pt)
        answer = get_answer(query_pt)

        if not answer:
            answer = buscar_no_wikipedia(query_pt)

        if not answer:
            answer = "Desculpe, não encontrei uma resposta abrangente sobre isso na base do Orkut e nem no Wikipédia."

        return intervention + answer if intervention else answer
    except Exception as e:
        logger.error("Exceção inesperada ao processar pergunta '%s': %s", user_text, e, exc_info=True)
        return "Desculpe, ocorreu um erro interno ao processar sua pergunta. Tente novamente."

# Sintetiza voz
def speak_response(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        speak_text = text.split('\n')[0]
        engine.say(speak_text)
        engine.runAndWait()
    except Exception as e:
        logger.error("Erro TTS ao sintetizar voz: %s", e, exc_info=True)

# ==========================================
# INTERFACE GRÁFICA TKINTER (ORKUT)
# ==========================================
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Inicialização do sistema — Chatbot Orkut")
    logger.info("Arquivo de log: %s", os.path.abspath(LOG_FILE))
    logger.info("=" * 50)

    root = tk.Tk()
    root.title("Orkut - Chatbot")
    root.geometry("950x750")
    root.configure(bg="#E5ECF9")

    header_frame = tk.Frame(root, bg="#C4D0EB", height=60)
    header_frame.pack(fill=tk.X)

    logo_label = tk.Label(header_frame, text="orkut",
                          font=("Arial", 28, "bold"),
                          fg="#D0028A", bg="#C4D0EB")
    logo_label.pack(side=tk.LEFT, padx=(20, 10), pady=10)

    menu_frame = tk.Frame(header_frame, bg="#C4D0EB")
    menu_frame.pack(side=tk.LEFT, padx=10, pady=22)

    menu_links = ["Início", "Perfil", "Página de recados", "Amigos", "Comunidades"]

    for i, link in enumerate(menu_links):
        lbl = tk.Label(menu_frame, text=link,
                       font=("Arial", 10),
                       fg="#0000CC", bg="#C4D0EB",
                       cursor="hand2")
        lbl.pack(side=tk.LEFT, padx=3)

        if i < len(menu_links) - 1:
            tk.Label(menu_frame, text="|",
                     font=("Arial", 10),
                     fg="#666666", bg="#C4D0EB").pack(side=tk.LEFT)

    search_frame = tk.Frame(header_frame, bg="#C4D0EB")
    search_frame.pack(side=tk.RIGHT, padx=20, pady=20)

    tk.Label(search_frame, text="buscar no orkut:",
             font=("Arial", 9),
             fg="#333333", bg="#C4D0EB").pack(side=tk.LEFT)

    tk.Entry(search_frame, width=20,
             highlightbackground="#CCCCCC",
             highlightthickness=1).pack(side=tk.LEFT, padx=5)

    # --------------------------
    # JANELA DO CHAT (SCROLLEDTEXT)
    # --------------------------
    chat_window = ScrolledText(
        root, wrap=tk.WORD,
        width=110, height=26,
        font=("Arial", 11),
        bg="#FFFFFF"
    )
    chat_window.pack(pady=20)

    # Configuração de estilos clássicos Orkut (Recados / Scraps)
    chat_window.tag_config("user_name", foreground="#0000CC", font=("Arial", 11, "bold"))
    chat_window.tag_config("meta_text", foreground="#666666", font=("Arial", 9, "italic"))
    chat_window.tag_config("user_msg", foreground="#333333", font=("Arial", 11))
    chat_window.tag_config("bot_name", foreground="#D0028A", font=("Arial", 11, "bold"))
    chat_window.tag_config("bot_msg", foreground="#000000", font=("Arial", 11))
    chat_window.tag_config("separator", foreground="#CCCCCC", font=("Arial", 9))
    chat_window.tag_config("info_tag", foreground="#888888", font=("Arial", 10, "italic"))

    chat_window.insert(tk.END, "Bem-vindo ao Chatbot do Orkut! Deixe um scrap abaixo para interagir com o bot.\n", "info_tag")
    chat_window.insert(tk.END, "="*85 + "\n", "separator")
    chat_window.configure(state="disabled")

    # --------------------------
    # CONTROLES DE ENTRADA
    # --------------------------
    input_frame = tk.Frame(root, bg="#E5ECF9")
    input_frame.pack(pady=10)

    entrada = tk.Entry(input_frame, width=70, font=("Arial", 12))
    entrada.pack(side=tk.LEFT, padx=10)

    # Envia mensagem
    def send_message(event=None):
        user_text = entrada.get().strip()
        if user_text == "":
            return "break"

        try:
            logger.info("Pergunta do usuário: %s", user_text)

            chat_window.config(state=tk.NORMAL)

            # Adiciona a mensagem do Usuário como Scrap
            chat_window.insert(tk.END, "Você ", "user_name")
            chat_window.insert(tk.END, "deixou um scrap:\n", "meta_text")
            chat_window.insert(tk.END, user_text + "\n", "user_msg")
            chat_window.insert(tk.END, "-" * 105 + "\n", "separator")

            # Obtém resposta abrangente do chatbot
            response = chatbot_response(user_text)
            logger.info("Resposta gerada: %s", response)

            # Adiciona a resposta do Bot como Scrap
            chat_window.insert(tk.END, "Chatbot do Orkut ", "bot_name")
            chat_window.insert(tk.END, "deixou um scrap:\n", "meta_text")
            chat_window.insert(tk.END, response + "\n", "bot_msg")
            chat_window.insert(tk.END, "-" * 105 + "\n", "separator")

            chat_window.see(tk.END)
            chat_window.config(state=tk.DISABLED)

            # Limpa campo
            entrada.delete(0, tk.END)

            # Inicia a fala em segundo plano
            threading.Thread(target=speak_response, args=(response,), daemon=True).start()
        except Exception as e:
            logger.error("Exceção inesperada na interface: %s", e, exc_info=True)

        return "break" # Evita que a tecla Enter insira uma quebra de linha indesejada

    # Vincula a tecla Enter e o botão para enviar
    entrada.bind("<Return>", send_message)

    btn = tk.Button(input_frame, text="Enviar Scrap",
                    command=send_message,
                    font=("Arial", 11, "bold"),
                    fg="#FFFFFF", bg="#D0028A",
                    activebackground="#A0016B", activeforeground="#FFFFFF",
                    cursor="hand2", borderwidth=0, padx=15, pady=5)
    btn.pack(side=tk.LEFT, padx=5)

    root.mainloop()
