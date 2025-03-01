import openai
import os
from dotenv import load_dotenv
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Načtení API klíče z .env souboru
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# Inicializace modelu pro embeddingy
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Znalostní báze - konkrétní odpovědi podle pohledu Filipa Dřímalky
knowledge_base = {
    "Filip Dřímalk": "Filip Dřímalka je český odborník na digitální transformaci a AI, autor knihy Budoucnost (ne)práce. Věří, že AI přináší nové příležitosti pro lidi, kteří jsou ochotní se učit a adaptovat na změny.",
    "Budoucnost (ne)práce": "Kniha se zaměřuje na dopady umělé inteligence na pracovní trh a budoucnost práce. Podle Filipa Dřímalky se musíme zaměřit na flexibilitu a neustálé vzdělávání, abychom se udrželi relevantní na trhu práce.",
    "Digitalizace": "Filip Dřímalka zdůrazňuje, že digitalizace není jen o technologiích, ale o změně myšlení. Firmy, které se naučí efektivně využívat AI, budou úspěšnější v moderním pracovním prostředí.",
    "transformace pracovního trhu": (
        "AI mění pracovní trh tím, že automatizuje rutinní úkoly. Některé profese zaniknou, zejména ty zaměřené na manuální a opakující se činnosti. Naopak vzniknou nové role v oblastech, kde AI doplňuje lidské schopnosti, například v kreativních profesích, strategickém rozhodování a technologických oborech."
    ),
    "AI a produktivita": (
        "AI umožňuje lidem soustředit se na práci s vyšší přidanou hodnotou. Zvyšuje produktivitu tím, že zpracovává velké objemy dat a nabízí prediktivní analýzy, které usnadňují rozhodování. Firmy by měly využít AI k optimalizaci procesů a efektivitě zaměstnanců."
    ),
    "celoživotní vzdělávání": (
        "Dřímalka zdůrazňuje, že jedinou jistotou v budoucnosti je neustálé vzdělávání. Lidé by měli investovat čas do osvojení nových dovedností, zejména v oblasti digitálních technologií, kritického myšlení a adaptability. AI nám pomůže učit se rychleji, ale rozhodující je otevřenost ke změnám."
    ),
    "implementace AI ve firmách": (
        "Firmy, které chtějí implementovat AI, by měly začít s jasnou strategií a definováním klíčových oblastí, kde může AI přinést největší přínos. Nejde o nahrazení lidí, ale o chytré využití AI k podpoře rozhodování a zvýšení efektivity." 
    ),
    "výzvy digitalizace": (
        "Největší výzvou digitalizace je změna firemní kultury a přijetí nových technologií zaměstnanci. Dřímalka doporučuje přístup založený na experimentech, postupném zavádění AI a vzdělávání zaměstnanců, aby se nebáli změn."
    )
}

# Seznam doporučených otázek
suggested_questions = [
    "Jak konkrétně AI mění pracovní trh a jaké profese zaniknou?",
    "Jaké konkrétní dovednosti budou v budoucnu nejvíce ceněné díky AI?",
    "Jak se můžu připravit na změny v pracovním trhu způsobené AI?",
    "Jaké strategie pro celoživotní vzdělávání doporučuje Filip Dřímalka?",
    "Jakým způsobem mohou firmy efektivně implementovat AI do svých procesů?",
    "Jaké jsou největší výzvy při digitalizaci a jak je překonat?"
]

# Konverze znalostní báze na embeddingy
text_chunks = list(knowledge_base.values())
embeddings = model.encode(text_chunks, convert_to_numpy=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Funkce pro vyhledávání nejrelevantnější odpovědi
def search_knowledge_base(query):
    query_embedding = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, 1)
    return text_chunks[indices[0][0]]

# Funkce chatbota kombinující FAISS a GPT-4
def ask_chatbot(prompt):
    relevant_info = search_knowledge_base(prompt)
    system_message = (
        "Jsi chatbot, který odpovídá na otázky ohledně budoucnosti práce, AI a digitální transformace z pohledu Filipa Dřímalky. "
        "Použij následující informace jako kontext pro svou odpověď:\n"
        f"{relevant_info}\n\n"
        "Pokud otázka nesouvisí s tímto tématem, odpověz neutrálně, že se zaměřuješ pouze na AI, budoucnost práce a digitální transformaci."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
def main():
    st.title("AI Průvodce pohledem Filipa Dřímalky na budoucnost (NE)práce a AI")
    st.write("Zeptej se na klíčové myšlenky z knihy a na jejího autora.")
    
    st.write("### Vyber otázku související s knihou:")
    selected_question = st.selectbox("", [""] + suggested_questions)
    
    user_input = st.text_input("Nebo napiš vlastní otázku, na kterou chceš znát odpověď:")
    
    if st.button("Odeslat"):
        query = user_input if user_input else selected_question
        if query:
            response = ask_chatbot(query)
            st.write("### Odpověď chatbota:")
            st.write(response)
        else:
            st.warning("Prosím, zadej otázku nebo vyber z nabídky.")

if __name__ == "__main__":
    main()

