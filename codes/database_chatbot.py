import difflib
import glob
import json
import os
import re

try:
    import numpy as np
except Exception:
    np = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


class DatabaseChatbot:
    def __init__(self, json_folder="herbaura_database"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.isabs(json_folder):
            self.json_folder = json_folder
        else:
            module_relative = os.path.join(base_dir, json_folder)
            self.json_folder = module_relative if os.path.isdir(module_relative) else json_folder

        # Core tables
        self.plants = []
        self.diseases = []
        self.remedies = []
        self.remedy_diseases = []
        self.plant_diseases = []
        self.precautions = []

        # Extra tables for hybrid mode
        self.medicinal_uses = []
        self.food_recipes = []
        self.disease_seasons = []

        # TF-IDF
        self.semantic_search_enabled = bool(np is not None and TfidfVectorizer is not None and cosine_similarity is not None)
        self.disease_vectorizer = None
        self.plant_vectorizer = None
        self.disease_matrix = None
        self.plant_matrix = None
        self.disease_index_map = []
        self.plant_index_map = []

        self.load_data()
        self.build_search_indexes()

    # -----------------------------
    # LOAD JSON TABLES
    # -----------------------------
    def load_data(self):
        json_files = sorted(glob.glob(os.path.join(self.json_folder, "*.json")))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in: {self.json_folder}")

        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            filename = os.path.basename(file)
            if isinstance(data, dict):
                data = list(data.values())
            if not isinstance(data, list):
                continue

            if filename == "plants.json":
                self.plants = data
            elif filename == "diseases.json":
                self.diseases = data
            elif filename == "remedies.json":
                self.remedies = data
            elif filename == "remedy_diseases.json":
                self.remedy_diseases = data
            elif filename == "plant_diseases.json":
                self.plant_diseases = data
            elif filename == "precautions.json":
                self.precautions = data
            elif filename == "medicinal_uses.json":
                self.medicinal_uses = data
            elif filename == "food_recipes.json":
                self.food_recipes = data
            elif filename == "disease_seasons.json":
                self.disease_seasons = data

        print("[OK] Database loaded successfully.")
        print("Plants:", len(self.plants))
        print("Diseases:", len(self.diseases))
        print("Remedies:", len(self.remedies))
        print("Medicinal Uses:", len(self.medicinal_uses))
        print("Offline Search Mode:", "TF-IDF+Fuzzy" if self.semantic_search_enabled else "Fuzzy-only")

    # -----------------------------
    # BUILD TF-IDF INDEXES
    # -----------------------------
    def build_search_indexes(self):
        self.disease_index_map = []
        self.plant_index_map = []
        if not self.semantic_search_enabled:
            return

        disease_docs = []
        for idx, disease in enumerate(self.diseases):
            text = (
                f"{disease.get('name', '')} "
                f"{disease.get('description', '')} "
                f"{disease.get('vernacular_names', '')}"
            ).strip()
            if text:
                disease_docs.append(text)
                self.disease_index_map.append(idx)

        if disease_docs:
            self.disease_vectorizer = TfidfVectorizer(stop_words="english")
            self.disease_matrix = self.disease_vectorizer.fit_transform(disease_docs)

        plant_docs = []
        for idx, plant in enumerate(self.plants):
            text = (
                f"{plant.get('common_name', '')} "
                f"{plant.get('scientific_name', '')} "
                f"{plant.get('description', '')} "
                f"{plant.get('ai_label', '')} "
                f"{plant.get('vernacular_name', '')}"
            ).strip()
            if text:
                plant_docs.append(text)
                self.plant_index_map.append(idx)

        if plant_docs:
            self.plant_vectorizer = TfidfVectorizer(stop_words="english")
            self.plant_matrix = self.plant_vectorizer.fit_transform(plant_docs)

    # -----------------------------
    # MAIN RESPONSE (HYBRID MODE)
    # -----------------------------
    def get_response(self, user_query):
        user_query = (user_query or "").strip()
        if not user_query:
            return (
                "I am here to help. Tell me a plant name, disease, "
                "or ask for remedies, medicinal uses, or precautions."
            )

        normalized = user_query.lower().strip()
        topic_guess = self._extract_topic(normalized)

        if self._is_greeting(normalized):
            return (
                "Hi, I am HerbAura AI and I work fully offline with your local database.\n"
                "You can ask me about plants, remedies, medicinal uses, and precautions."
            )

        if self._is_thanks(normalized):
            return "You are welcome. If you want, ask me about another plant, remedy, or precaution."

        if self._is_small_talk(normalized):
            return "I am doing well, thank you. Tell me your symptom or plant/remedy request and I will help."

        topic_query = self._strip_intent_words(normalized)
        intent = self._detect_intent(normalized)
        lookup_query = topic_query or topic_guess or normalized

        if any(x in normalized for x in ["hair oil", "hairoil", "wellness", "haircare", "hair care"]):
            wellness_response = self.search_wellness_remedies(lookup_query)
            if wellness_response:
                return wellness_response
            return self._no_verified_data_message("wellness/hair-oil information")

        negated = self._detect_negated_disease(normalized)
        if negated:
            return (
                f"Understood, not {negated}.\n"
                "I can only answer conditions available in my local database right now: "
                f"{', '.join([d.get('name', '') for d in self.diseases])}."
            )

        if intent in {"general", "remedy", "season"}:
            triage_response = self._symptom_triage_response(lookup_query)
            if triage_response:
                return triage_response

        if intent == "use":
            plant = self.search_plant(lookup_query, return_record=True)
            if plant:
                return self.format_medicinal_use_response(plant)

        if intent == "precaution":
            plant = self.search_plant(lookup_query, return_record=True)
            if plant:
                return self.format_precaution_response(plant)

        if intent == "recipe":
            plant = self.search_plant(lookup_query, return_record=True)
            if plant:
                return self.format_recipe_response(plant)
            return self._no_verified_data_message("recipe information")

        if intent == "season":
            disease = self.search_disease(lookup_query, return_record=True)
            if disease:
                return self.format_disease_season_response(disease)

        if intent == "remedy":
            keyword_disease = self._match_disease_from_symptom_text(lookup_query)
            if keyword_disease:
                return self.format_disease_response(keyword_disease)
            disease = self.search_disease(lookup_query, return_record=True)
            if disease:
                return self.format_disease_response(disease)
            plant = self.search_plant(lookup_query, return_record=True)
            if plant:
                return self.format_plant_remedy_response(plant)

        keyword_disease = self._match_disease_from_symptom_text(lookup_query)
        if keyword_disease:
            return self.format_disease_response(keyword_disease)

        disease_response = self.search_disease(lookup_query)
        if disease_response:
            return disease_response

        plant_response = self.search_plant(lookup_query)
        if plant_response:
            return plant_response

        suggestion = self.spell_suggestion(lookup_query)
        if suggestion:
            if suggestion["type"] == "plant":
                matched = self.get_plant_by_name(suggestion["value"])
                if matched:
                    return self.format_plant_response(
                        matched,
                        prefixed_note=f"I matched your query to '{suggestion['value']}'.",
                    )
            if suggestion["type"] == "disease":
                matched = self.get_disease_by_name(suggestion["value"])
                if matched:
                    return self.format_disease_response(
                        matched,
                        prefixed_note=f"I matched your query to '{suggestion['value']}'.",
                    )
            return f"Did you mean '{suggestion['value']}'?"

        return (
            "I could not find a verified match in my current database.\n"
            "Please try a plant name, disease name, or ask for uses, remedies, and precautions.\n"
            "If this is related to health concerns, consult a doctor for proper guidance."
        )

    # -----------------------------
    # DISEASE SEARCH
    # -----------------------------
    def search_disease(self, query, return_record=False):
        exact = self.get_disease_by_name(query)
        if exact:
            return exact if return_record else self.format_disease_response(exact)

        close = self._best_close_match(query, [d.get("name", "") for d in self.diseases], cutoff=0.80)
        if close:
            match = self.get_disease_by_name(close)
            if match and self._has_strong_disease_anchor(query, match.get("name", "")):
                return match if return_record else self.format_disease_response(match)

        if (not self.semantic_search_enabled) or self.disease_matrix is None or self.disease_vectorizer is None:
            return None

        query_vec = self.disease_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.disease_matrix)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[0][best_idx])

        if best_score < 0.30:
            return None

        disease = self.diseases[self.disease_index_map[best_idx]]
        if not self._has_strong_disease_anchor(query, disease.get("name", "")):
            return None
        return disease if return_record else self.format_disease_response(disease)

    # -----------------------------
    # PLANT SEARCH
    # -----------------------------
    def search_plant(self, query, return_record=False):
        exact = self.get_plant_by_name(query)
        if exact:
            return exact if return_record else self.format_plant_response(exact)

        all_names = []
        for plant in self.plants:
            all_names.extend(
                [
                    plant.get("common_name", ""),
                    plant.get("scientific_name", ""),
                    plant.get("ai_label", ""),
                    plant.get("vernacular_name", ""),
                ]
            )
        close = self._best_close_match(query, all_names, cutoff=0.76)
        if close:
            match = self.get_plant_by_name(close)
            if match:
                return match if return_record else self.format_plant_response(match)

        if (not self.semantic_search_enabled) or self.plant_matrix is None or self.plant_vectorizer is None:
            return None

        query_vec = self.plant_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.plant_matrix)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[0][best_idx])

        if best_score < 0.32:
            return None

        plant = self.plants[self.plant_index_map[best_idx]]
        if not self._has_strong_plant_anchor(query, plant):
            return None
        return plant if return_record else self.format_plant_response(plant)

    # -----------------------------
    # FORMATTERS
    # -----------------------------
    def format_disease_response(self, disease, prefixed_note=None):
        disease_id = disease.get("id")
        response = ""
        if prefixed_note:
            response += f"{prefixed_note}\n\n"

        response += f"{disease.get('name', 'Disease').title()}\n"
        response += f"\nDescription:\n{disease.get('description', '')}\n"

        remedies = self.get_remedies_for_disease_id(disease_id)
        if remedies:
            response += "\nRecommended Remedies:\n"
            for remedy in remedies:
                response += f"\n- {remedy.get('title', 'Remedy')}\n"
                response += f"{self._clean_text(remedy.get('instructions', ''))}\n"
                plant = next((p for p in self.plants if p.get("id") == remedy.get("plant_id")), None)
                if plant:
                    response += (
                        f"Plant Used: {plant.get('common_name', '')} "
                        f"({plant.get('scientific_name', '')})\n"
                    )
        else:
            response += "\nI do not currently have linked remedies for this disease."

        response += "\n\nI hope this helps. This is not medical advice. Consult a doctor if symptoms persist."
        return response

    def format_plant_response(self, plant, prefixed_note=None):
        plant_id = plant.get("id")
        response = ""
        if prefixed_note:
            response += f"{prefixed_note}\n\n"

        response += f"{plant.get('common_name', 'Plant').title()}\n"
        response += f"Scientific Name: {plant.get('scientific_name', '')}\n"
        response += f"AI Label: {plant.get('ai_label', '')}\n"
        response += f"\nDescription:\n{plant.get('description', '')}\n"

        uses = [u.get("use_text", "") for u in self.medicinal_uses if u.get("plant_id") == plant_id]
        if uses:
            response += "\nMedicinal Uses:\n"
            for use in uses[:3]:
                response += f"- {use}\n"

        precautions = [p for p in self.precautions if p.get("plant_id") == plant_id]
        if precautions:
            response += "\nPrecautions:\n"
            for precaution in precautions:
                response += f"- {precaution.get('precaution_text', '')}\n"

        return response

    def format_medicinal_use_response(self, plant):
        plant_id = plant.get("id")
        uses = [u.get("use_text", "") for u in self.medicinal_uses if u.get("plant_id") == plant_id]
        if not uses:
            return self._no_verified_data_message(
                f"medicinal use information for {plant.get('common_name', 'this plant')}"
            )
        response = f"Medicinal uses of {plant.get('common_name', 'this plant')}:\n"
        for use in uses:
            response += f"- {use}\n"
        return response

    def format_precaution_response(self, plant):
        plant_id = plant.get("id")
        items = [p.get("precaution_text", "") for p in self.precautions if p.get("plant_id") == plant_id]
        if not items:
            return self._no_verified_data_message(
                f"precaution details for {plant.get('common_name', 'this plant')}"
            )
        response = f"Precautions for {plant.get('common_name', 'this plant')}:\n"
        for item in items:
            response += f"- {item}\n"
        return response

    def format_recipe_response(self, plant):
        plant_id = plant.get("id")
        recipes = [r for r in self.food_recipes if r.get("plant_id") == plant_id]
        if not recipes:
            return self._no_verified_data_message(
                f"recipe details for {plant.get('common_name', 'this plant')}"
            )
        response = f"Recipes using {plant.get('common_name', 'this plant')}:\n"
        for recipe in recipes:
            response += f"\n- {recipe.get('title', 'Recipe')}\n"
            if recipe.get("ingredients"):
                response += f"  Ingredients: {recipe.get('ingredients')}\n"
            if recipe.get("instructions"):
                response += f"  Instructions: {recipe.get('instructions')}\n"
        return response

    def format_plant_remedy_response(self, plant):
        plant_id = plant.get("id")
        disease_ids = [x.get("disease_id") for x in self.plant_diseases if x.get("plant_id") == plant_id]
        disease_ids = [x for x in disease_ids if x is not None]
        if not disease_ids:
            return self._no_verified_data_message(
                f"linked remedy details for {plant.get('common_name', 'this plant')}"
            )

        response = f"Remedy links for {plant.get('common_name', 'this plant')}:\n"
        for disease_id in disease_ids:
            disease = next((d for d in self.diseases if d.get("id") == disease_id), None)
            if not disease:
                continue
            response += f"\nDisease: {disease.get('name', 'Unknown')}\n"
            remedies = self.get_remedies_for_disease_id(disease_id)
            if remedies:
                for remedy in remedies:
                    response += f"- {remedy.get('title', 'Remedy')}: {self._clean_text(remedy.get('instructions', ''))}\n"
            else:
                response += "- No specific remedy entry found.\n"
        return response

    def format_disease_season_response(self, disease):
        disease_id = disease.get("id")
        entries = [s for s in self.disease_seasons if s.get("disease_id") == disease_id]
        if not entries:
            return self._no_verified_data_message(
                f"seasonal information for {disease.get('name', 'this disease')}"
            )
        response = f"Seasonal notes for {disease.get('name', 'this disease')}:\n"
        for item in entries:
            season = item.get("season_name") or item.get("season") or "Unknown Season"
            note = item.get("notes") or item.get("description") or ""
            response += f"- {season}"
            if note:
                response += f": {note}"
            response += "\n"
        return response

    def format_wellness_remedies(self, remedies):
        if not remedies:
            return None
        response = "Here are verified wellness remedies available in your local database:\n"
        for remedy in remedies:
            response += f"\n- {remedy.get('title', 'Wellness Remedy')}\n"
            if remedy.get("sub_category"):
                response += f"  Type: {remedy.get('sub_category')}\n"
            instructions = self._clean_text(remedy.get("instructions", ""))
            if instructions:
                response += f"  Procedure:\n{instructions}\n"
            notes = self._clean_text(remedy.get("notes", ""))
            if notes:
                response += f"  Notes: {notes}\n"
        response += "\nPlease do a patch test first and consult a qualified doctor for persistent issues."
        return response

    # -----------------------------
    # HELPERS
    # -----------------------------
    def get_plant_by_name(self, query):
        q = self._normalize(query)
        for plant in self.plants:
            candidates = [
                plant.get("common_name", ""),
                plant.get("scientific_name", ""),
                plant.get("ai_label", ""),
                plant.get("vernacular_name", ""),
            ]
            if any(q and q == self._normalize(c) for c in candidates):
                return plant
        return None

    def get_disease_by_name(self, query):
        q = self._normalize(query)
        for disease in self.diseases:
            if q and q == self._normalize(disease.get("name", "")):
                return disease
        return None

    def get_remedies_for_disease_id(self, disease_id):
        linked = [r for r in self.remedy_diseases if r.get("disease_id") == disease_id]
        remedy_ids = [r.get("remedy_id") for r in linked]
        return [r for r in self.remedies if r.get("id") in remedy_ids]

    def search_wellness_remedies(self, query):
        q = self._normalize(query)
        wellness = [
            r for r in self.remedies
            if "wellness" in self._normalize(r.get("category", ""))
            or "hair oil" in self._normalize(r.get("sub_category", ""))
        ]
        if not wellness:
            return None

        if any(x in q for x in ["hair", "oil", "wellness", "procedure", "make", "prepare"]):
            return self.format_wellness_remedies(wellness)

        scored = []
        for remedy in wellness:
            text = " ".join([
                remedy.get("title", ""),
                remedy.get("sub_category", ""),
                remedy.get("notes", ""),
                remedy.get("instructions", ""),
            ])
            score = difflib.SequenceMatcher(None, q, self._normalize(text)).ratio()
            scored.append((score, remedy))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] >= 0.28:
            return self.format_wellness_remedies([scored[0][1]])
        return None

    def spell_suggestion(self, query):
        candidates = []
        for disease in self.diseases:
            name = disease.get("name", "")
            if name:
                candidates.append(("disease", name))
        for plant in self.plants:
            for value in [
                plant.get("common_name", ""),
                plant.get("scientific_name", ""),
                plant.get("ai_label", ""),
                plant.get("vernacular_name", ""),
            ]:
                if value:
                    candidates.append(("plant", value))

        normalized_values = {self._normalize(value): (ctype, value) for ctype, value in candidates}
        normalized_query = self._normalize(query)
        candidates_to_try = [normalized_query] + normalized_query.split()
        best = []
        for candidate in candidates_to_try:
            if not candidate:
                continue
            best = difflib.get_close_matches(candidate, list(normalized_values.keys()), n=1, cutoff=0.72)
            if best:
                break
        if best:
            ctype, value = normalized_values[best[0]]
            return {"type": ctype, "value": value}
        return None

    def _best_close_match(self, query, values, cutoff=0.72):
        cleaned = [v for v in values if isinstance(v, str) and v.strip()]
        if not cleaned:
            return None
        norm_map = {self._normalize(v): v for v in cleaned}
        best = difflib.get_close_matches(self._normalize(query), list(norm_map.keys()), n=1, cutoff=cutoff)
        return norm_map[best[0]] if best else None

    def _normalize(self, text):
        if text is None:
            return ""
        return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

    def _detect_intent(self, query):
        if any(x in query for x in ["remedy", "remedies", "cure", "treat", "treatment"]):
            return "remedy"
        if any(x in query for x in ["medicinal use", "use", "benefit", "uses"]):
            return "use"
        if any(x in query for x in ["precaution", "safety", "safe", "side effect"]):
            return "precaution"
        if any(x in query for x in ["recipe", "food", "eat", "cooking"]):
            return "recipe"
        if any(x in query for x in ["season", "seasonal", "monsoon", "summer", "winter"]):
            return "season"
        if any(x in query for x in ["how to", "procedure", "steps", "make", "prepare"]):
            return "recipe"
        return "general"

    def _is_greeting(self, query):
        greetings = ["hi", "hello", "hey", "good morning", "good evening"]
        if query in greetings:
            return True
        parts = self._normalize(query).split()
        if len(parts) == 1:
            token = parts[0]
            for g in ["hi", "hey", "hello"]:
                if difflib.SequenceMatcher(None, token, g).ratio() >= 0.75:
                    return True
        return False

    def _is_thanks(self, query):
        q = self._normalize(query)
        if not q:
            return False
        return any(x in q for x in ["thanks", "thank you", "thank u", "thx"])

    def _is_small_talk(self, query):
        q = self._normalize(query)
        if not q:
            return False
        return any(
            x in q
            for x in ["how do you do", "how are you", "how r u", "how do u do", "wassup", "what s up"]
        )

    def _extract_topic(self, query):
        tokens = self._normalize(query).split()
        filler = {
            "do", "you", "know", "about", "u", "knwo", "can", "please", "tell", "me", "i", "am",
            "having", "has", "have", "got", "my", "is", "are", "the", "a", "an", "of", "on", "by",
        }
        core = [t for t in tokens if t not in filler]
        return " ".join(core).strip()

    def _is_vague_symptom_query(self, query):
        tokens = self._normalize_and_fuzzy_tokens(query)
        if not tokens:
            return False

        vague_symptoms = {
            "pain",
            "ache",
            "aching",
            "shiver",
            "shivering",
            "chills",
            "fever",
            "discomfort",
            "hurt",
        }
        specific_clues = {
            "spider",
            "insect",
            "bee",
            "wasp",
            "sting",
            "bite",
            "skin",
            "allergy",
            "rash",
            "wound",
            "cut",
            "menstrual",
            "period",
        }

        token_set = set(tokens)
        has_vague = bool(token_set & vague_symptoms)
        has_specific = bool(token_set & specific_clues)
        return has_vague and not has_specific

    def _clarify_symptom_message(self, query):
        q = self._normalize(query)
        if any(x in q for x in ["shiver", "chill", "fever"]):
            return (
                "I need a bit more detail to identify the right condition.\n"
                "Do you also have any of these: insect/bee sting, spider bite, skin rash, or wound/cut?\n"
                "If you have high fever, breathing difficulty, severe weakness, or persistent shivering, consult a doctor urgently."
            )

        return (
            "I can help narrow this down. Please tell me one specific detail:\n"
            "- Was it an insect/bee/wasp sting?\n"
            "- Was it a spider bite?\n"
            "- Is it a skin rash/allergy?\n"
            "- Is it a wound or cut?\n"
            "- Is it menstrual pain?\n"
            "If pain is severe or persistent, consult a doctor immediately."
        )

    def _match_disease_from_symptom_text(self, query):
        tokens = self._normalize_and_fuzzy_tokens(query)
        token_set = set(tokens)

        bite_words = {"bite", "bitten", "bit", "sting", "stung"}
        if "spider" in token_set and token_set & bite_words:
            return self.get_disease_by_name("Spider Bite")
        if ("insect" in token_set or "bee" in token_set or "wasp" in token_set) and token_set & bite_words:
            return self.get_disease_by_name("Insect Sting (Bee / Wasp)")
        if "allergy" in token_set or ({"skin"} <= token_set and token_set & {"rash", "itch", "itching", "allergic"}):
            return self.get_disease_by_name("Skin Allergy")
        if token_set & {"wound", "wounds", "cut", "cuts", "bleeding"}:
            return self.get_disease_by_name("Wounds and Cuts")
        if ({"menstrual", "period"} & token_set) and (token_set & {"pain", "cramp", "cramps"}):
            return self.get_disease_by_name("Menstrual pain")
        return None

    def _normalize_and_fuzzy_tokens(self, text):
        tokens = self._normalize(text).split()
        disease_vocab = {
            "insect", "bee", "wasp", "sting", "bite", "bitten", "spider", "skin",
            "allergy", "allergic", "rash", "itch", "itching", "wound", "wounds",
            "cut", "cuts", "bleeding", "menstrual", "period", "pain", "cramp", "cramps",
            "swelling", "irritation", "burning", "redness", "shiver", "shivering",
            "chills", "fever", "weakness", "vomit", "nausea", "dizzy", "headache",
        }
        corrected = []
        for token in tokens:
            best = difflib.get_close_matches(token, list(disease_vocab), n=1, cutoff=0.75)
            corrected.append(best[0] if best else token)
        return corrected

    def _symptom_triage_response(self, query):
        if not self._looks_like_symptom_query(query):
            return None

        direct_match = self._match_disease_from_symptom_text(query)
        if direct_match:
            return self.format_disease_response(direct_match)

        ranked = self._rank_disease_candidates(query)
        if not ranked:
            return (
                "I could not map this symptom to a verified condition in my current database.\n"
                f"Available conditions: {', '.join([d.get('name', '') for d in self.diseases])}.\n"
                "If symptoms are serious or worsening, consult a doctor immediately."
            )

        top = ranked[0]
        top_score = top["score"]
        margin = top_score - (ranked[1]["score"] if len(ranked) > 1 else 0.0)

        if top_score >= 0.52 and margin >= 0.10:
            return self.format_disease_response(
                top["disease"],
                prefixed_note="Based on your symptoms, this is the closest verified match in my database.",
            )

        candidates = ranked[:3]
        options = ", ".join([c["disease"].get("name", "Unknown") for c in candidates])
        prompt = "I need one more detail to rule out the correct condition."
        if any("bite" in c["anchors"] or "sting" in c["anchors"] for c in candidates):
            prompt += "\nWas this from an insect/bee/wasp sting, or a spider bite?"
        elif any("skin" in c["anchors"] or "rash" in c["anchors"] or "allergy" in c["anchors"] for c in candidates):
            prompt += "\nIs there visible skin rash/itching/allergic reaction?"
        elif any("wound" in c["anchors"] or "cut" in c["anchors"] for c in candidates):
            prompt += "\nIs there an open wound or cut?"
        else:
            prompt += "\nPlease mention location, trigger, and associated signs (rash, swelling, wound, bite, menstrual cycle)."

        return (
            f"{prompt}\n"
            f"Closest verified possibilities: {options}.\n"
            "If symptoms are severe, sudden, or worsening, consult a doctor immediately."
        )

    def _rank_disease_candidates(self, query):
        q_tokens = self._symptom_tokens(query)
        if not q_tokens:
            return []

        query_text = " ".join(q_tokens)
        ranked = []
        for disease in self.diseases:
            d_name = disease.get("name", "")
            d_desc = disease.get("description", "")
            d_vern = disease.get("vernacular_names", "")
            d_text = f"{d_name} {d_desc} {d_vern}"
            d_tokens = self._symptom_tokens(d_text)
            if not d_tokens:
                continue

            overlap = q_tokens & d_tokens
            overlap_score = len(overlap) / max(1, len(q_tokens))
            containment_score = len(overlap) / max(1, len(d_tokens))
            text_score = difflib.SequenceMatcher(None, query_text, " ".join(d_tokens)).ratio()
            score = (0.60 * overlap_score) + (0.20 * containment_score) + (0.20 * text_score)

            ranked.append(
                {
                    "disease": disease,
                    "score": score,
                    "anchors": overlap,
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return [r for r in ranked if r["score"] >= 0.18]

    def _symptom_tokens(self, text):
        tokens = self._normalize_and_fuzzy_tokens(text)
        stop = {
            "i", "am", "have", "having", "has", "had", "the", "a", "an", "my", "me", "you",
            "do", "did", "is", "are", "was", "were", "to", "for", "of", "on", "in", "with",
            "very", "little", "bit", "from", "after", "before", "since", "and", "or", "but",
            "want", "need", "help", "feel", "feeling", "body", "there", "it", "this", "that",
        }
        return {t for t in tokens if t and t not in stop}

    def _looks_like_symptom_query(self, query):
        q = self._normalize(query)
        if not q:
            return False
        symptom_markers = {
            "pain", "swelling", "itch", "itching", "rash", "allergy", "wound", "cut",
            "bite", "bitten", "sting", "stung", "fever", "chills", "shiver", "cramp",
            "bleeding", "irritation", "burning", "nausea", "vomit", "dizzy", "weakness",
        }
        tokens = set(self._normalize_and_fuzzy_tokens(q))
        phrase_markers = ["i have", "i am having", "i feel", "symptom", "bitten by", "stung by"]
        return bool(tokens & symptom_markers) or any(p in q for p in phrase_markers)

    def _detect_negated_disease(self, query):
        if not any(x in query for x in ["not ", "no "]):
            return None
        for disease in self.diseases:
            name = disease.get("name", "")
            if not name:
                continue
            if self._normalize(name) in self._normalize(query):
                return name
        return None

    def _has_strong_disease_anchor(self, query, disease_name):
        q_tokens = set(self._normalize(query).split())
        d_tokens = set(self._normalize(disease_name).split())
        generic = {"pain", "allergy", "bite", "sting", "wounds", "cuts", "and"}
        d_tokens = {t for t in d_tokens if t not in generic}
        return bool(q_tokens & d_tokens)

    def _has_strong_plant_anchor(self, query, plant):
        q_tokens = set(self._normalize(query).split())
        if not q_tokens:
            return False
        generic = {
            "plant", "herb", "leaf", "leaves", "tree", "medicine", "medicinal",
            "unknown", "about", "know", "tell", "me", "do", "you", "u", "what",
        }
        q_tokens = {t for t in q_tokens if t not in generic}
        if not q_tokens:
            return False

        plant_text = " ".join(
            [
                plant.get("common_name", ""),
                plant.get("scientific_name", ""),
                plant.get("ai_label", ""),
                plant.get("vernacular_name", ""),
            ]
        )
        p_tokens = set(self._normalize(plant_text).split())
        return bool(q_tokens & p_tokens)

    def _clean_text(self, text):
        return str(text or "").replace("\\n", "\n").strip()

    def _no_verified_data_message(self, topic):
        return (
            f"I found the request, but verified {topic} is not available in my current database.\n"
            "For safe and accurate medical guidance, please consult a qualified doctor."
        )

    def _strip_intent_words(self, query):
        words_to_remove = {
            "remedy",
            "remedies",
            "cure",
            "treat",
            "treatment",
            "medicinal",
            "use",
            "uses",
            "benefits",
            "benefit",
            "precaution",
            "precautions",
            "safe",
            "safety",
            "recipe",
            "recipes",
            "food",
            "eat",
            "cooking",
            "season",
            "seasonal",
            "during",
            "for",
            "of",
            "about",
            "tell",
            "me",
            "what",
            "is",
            "the",
        }
        tokens = [t for t in self._normalize(query).split() if t not in words_to_remove]
        return " ".join(tokens).strip()
