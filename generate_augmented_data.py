import os
import sys
import re
import unicodedata
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def clean_bangla(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    DATA_DIR = "./data"
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "validation.csv")
    out_path = os.path.join(DATA_DIR, "train_augmented.csv")

    train_df = pd.read_csv(train_path, encoding="utf-8")
    val_df = pd.read_csv(val_path, encoding="utf-8")
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Loaded original train+val: {len(full_df)} samples")

    target_cols = [
        "is_toxic", "implicit_toxicity", "sarcasm",
        "contains_profanity", "threatening", "identity_attack", "sexual_content"
    ]

    full_df["clean_text"] = full_df["text"].apply(clean_bangla)
    full_df["is_augmented"] = 0

    augmented_rows = []

    # =========================================================================
    # 1. Targeted Bengali Threatening Samples (Lift from 108 to ~250)
    # =========================================================================
    threat_sentences = [
        # Direct violent & death threats
        ("তোকে রাস্তায় পেলে পিটিয়ে মেরে ফেলব!", 1, 0, 0, 0, 1, 0, 0),
        ("তোর মাথা কেটে নদীতে ভাসিয়ে দেব, সাবধান হয়ে যা!", 1, 0, 0, 0, 1, 0, 0),
        ("বাড়ির বাইরে বের হ, তোকে আজই শেষ করে ফেলব!", 1, 0, 0, 0, 1, 0, 0),
        ("তোর মতো কুলাঙ্গারকে বাঁচিয়ে রাখা পাপ, তোকে খুন করব!", 1, 0, 0, 1, 1, 0, 0),
        ("তোর লাশ কেউ খুঁজে পাবে না, এমন জায়গায় পুতে রাখব!", 1, 0, 0, 0, 1, 0, 0),
        ("তোর কবর খোঁড়া হয়ে গেছে, শুধু তোর যাওয়ার বাকি!", 1, 0, 0, 0, 1, 0, 0),
        ("তোকে আর তোর পরিবারকে একেবারে শেষ করে দেব!", 1, 0, 0, 0, 1, 0, 0),
        ("তোর দুই চোখ উপড়ে ফেলব যদি আর একবার এই কথা বলিস!", 1, 0, 0, 0, 1, 0, 0),
        ("হাত-পা ভেঙে তোকে চিরজীবনের জন্য পঙ্গু বানিয়ে দেব!", 1, 0, 0, 0, 1, 0, 0),
        ("চামড়া তুলে লবণ মেখে দেব, বেশি বাড়াবাড়ি করিস না!", 1, 0, 0, 0, 1, 0, 0),
        ("তোকে এমন মার মারব যে হাসপাতালে যাওয়ার সময় পাবি না!", 1, 0, 0, 0, 1, 0, 0),
        ("তোর বুকের উপর চড়ে তোকে জবাই করব, কুত্তার বাচ্চা!", 1, 0, 0, 1, 1, 0, 0),
        ("তোর জিহ্বা কেটে কুত্তা দিয়ে খাওয়াব বেশি কথা বললে!", 1, 0, 0, 1, 1, 0, 0),
        ("গলায় পাড়া দিয়ে তোকে শেষ করে ফেলব বদমাশ!", 1, 0, 0, 1, 1, 0, 0),
        ("তোর হাড়গোড় সব গুঁড়ো করে ফেলব রাস্তায় পেলে!", 1, 0, 0, 0, 1, 0, 0),
        ("তোর পেটে ছুরি ঢুকিয়ে নাড়িভুঁড়ি বের করে আনব!", 1, 0, 0, 0, 1, 0, 0),
        ("তোকে জীবিত পুড়িয়ে মারব, তোর রক্ষা নেই!", 1, 0, 0, 0, 1, 0, 0),
        ("তোর রক্ত দিয়ে হোলি খেলব তুই শুধু সামনে আয়!", 1, 0, 0, 0, 1, 0, 0),
        ("তোর গর্দান নামিয়ে দেব এক কোপে, বেশি পকপক করিস না!", 1, 0, 0, 0, 1, 0, 0),
        ("তোকে গাড়ি চাপা দিয়ে মেরে ফেলব যেন কেউ টেরই না পায়!", 1, 0, 0, 0, 1, 0, 0),

        # Extortion, property & family threats
        ("তোর ঘরবাড়ি পেট্রল দিয়ে জ্বালিয়ে ছারখার করে দেব!", 1, 0, 0, 0, 1, 0, 0),
        ("তোর দোকান লুটপাট করে তোকে পথের ভিখারি বানাব!", 1, 0, 0, 0, 1, 0, 0),
        ("টাকা না দিলে তোর সন্তানকে আর জীবিত দেখতে পাবি না!", 1, 0, 0, 0, 1, 0, 0),
        ("কালকের মধ্যে এলাকা না ছাড়লে তোর বংশ নির্বংশ করে দেব!", 1, 0, 0, 0, 1, 0, 0),
        ("তোর মা-বোনকে তুলে নিয়ে যাব যদি বাড়াবাড়ি করিস!", 1, 0, 0, 0, 1, 0, 1),
        ("তোর ব্যবসা পুড়িয়ে দেব, কোটি টাকার ক্ষতি করে ছাড়ব!", 1, 0, 0, 0, 1, 0, 0),
        ("পুলিশের কাছে গেলে তোকে গুলি করে মাথার খুলি উড়িয়ে দেব!", 1, 0, 0, 0, 1, 0, 0),
        ("তোর অফিসে গিয়ে তোকে সবার সামনে গুলি করব!", 1, 0, 0, 0, 1, 0, 0),
        ("আজ রাতের মধ্যে টাকা রেডি রাখিস, নাহলে তোর মৃত্যু নিশ্চিত!", 1, 0, 0, 0, 1, 0, 0),
        ("তোকে এই শহরে আর কোনোদিন ঢুকতে দেব না, কেটে টুকরো করব!", 1, 0, 0, 0, 1, 0, 0),

        # Banglish threats
        ("toke rastay pele petabo, khub bhalo bhabe!", 1, 0, 0, 0, 1, 0, 0),
        ("toke jaan theke mere phelbo bujhis!", 1, 0, 0, 0, 1, 0, 0),
        ("toke kete nodite bhashiye debo kuttar bacha!", 1, 0, 0, 1, 1, 0, 0),
        ("tor bari jwaliye debo shobai k shonai!", 1, 0, 0, 0, 1, 0, 0),
        ("toke khotom kore debo ekdom dekhe nish!", 1, 0, 0, 0, 1, 0, 0),
    ]

    for i, item in enumerate(threat_sentences):
        text, tox, imp, sarc, prof, threat, ident, sex = item
        augmented_rows.append({
            "ID": f"aug_threat_gen_{i}",
            "text": text,
            "clean_text": clean_bangla(text),
            "is_toxic": tox,
            "implicit_toxicity": imp,
            "sarcasm": sarc,
            "contains_profanity": prof,
            "threatening": threat,
            "identity_attack": ident,
            "sexual_content": sex,
            "is_augmented": 1
        })

    # =========================================================================
    # 2. Targeted Bengali Sarcasm & Implicit Toxicity
    # =========================================================================
    sarcasm_implicit_sentences = [
        ("আপনার মতো দূরদর্শী নেতা থাকলে দেশের ধ্বংস হতে আর বেশিদিন লাগবে না!", 1, 1, 1, 0, 0, 0, 0),
        ("বাহ চমৎকার কাজ! দেশের বারোটা বাজিয়ে এখন পুরস্কার দাবি করছেন!", 1, 1, 1, 0, 0, 0, 0),
        ("এত সুন্দর করে মিথ্যে কথা বলতে পারাটাও একটা আর্ট, আপনি সেই আর্টের রাজা!", 1, 1, 1, 0, 0, 0, 0),
        ("আপনার সততা দেখে তো দেশের চোর-ডাকাতরাও লজ্জা পেয়ে পেশা ছেড়ে দেবে!", 1, 1, 1, 0, 0, 0, 0),
        ("একেই বলে আসল উন্নয়ন, মানুষ না খেয়ে রাস্তায় ঘুমাচ্ছে আর আপনারা বিলবোর্ড লাগাচ্ছেন!", 1, 1, 1, 0, 0, 0, 0),
        ("আপনার মেধা দেখলে আইনস্টাইন নিজেই নিজের মাথায় হাত দিয়ে বসে পড়তেন!", 1, 1, 1, 0, 0, 0, 0),
        ("মাশাআল্লাহ! কি সুন্দরভাবে জনগণের কোটি কোটি টাকা নিজের পকেটে ঢুকিয়ে নিলেন!", 1, 1, 1, 0, 0, 0, 0),
        ("আপনার নীতিবাক্য শুনে চোখের পানি ধরে রাখা মুশকিল, অথচ নিজেই সব কেলেঙ্কারির মূল!", 1, 1, 1, 0, 0, 0, 0),
        ("বাহ! কি অসাধারণ যুক্তি, নিজের চুরি ঢাকতে এখন ধর্মের দোহাই দিচ্ছেন!", 1, 1, 1, 0, 0, 0, 0),
        ("সাধু সাজা বন্ধ করুন, আপনার মুখোশ অনেক আগেই খসে পড়েছে!", 1, 1, 0, 0, 0, 0, 0),
        ("এত তেল মারা বক্তব্য কোথা থেকে শিখে আসেন ভাই? চামচামিরও তো একটা সীমা থাকে!", 1, 1, 0, 0, 0, 0, 0),
        ("অস্কার পুরস্কারটা এবার আপনাদের দেওয়া উচিত, কি দারুণ অভিনয়ের নাটক দেখালেন!", 1, 1, 1, 0, 0, 0, 0),
        ("দেশের টাকা বিদেশে পাচার করে এখন দেশপ্রেমের লম্বা লম্বা লেকচার শোনাচ্ছেন!", 1, 1, 1, 0, 0, 0, 0),
        ("আপনার মতো সুবিধাবাদী পরজীবী সমাজে খুব কমই দেখা যায়!", 1, 1, 0, 0, 0, 0, 0),
        ("এত মিষ্টি কথায় বিষ মাখানো মানুষ জীবনে প্রথম দেখলাম!", 1, 1, 1, 0, 0, 0, 0),
        ("আপনার মতো পণ্ডিত ব্যক্তিদের মুখে চুনকালি পড়াটাই স্বাভাবিক!", 1, 1, 1, 0, 0, 0, 0),
        ("বাহ কি সুন্দর সিস্টেম, গরিবের পেটে লাথি মেরে ধনীদের ট্যাক্স মাফ!", 1, 1, 1, 0, 0, 0, 0),
        ("নিজের অপকর্মের গন্ধ ঢাকতে এখন সেন্ট মারছেন আপনি!", 1, 1, 1, 0, 0, 0, 0),
        ("যাক অবশেষে জানা গেল চোরের মায়ের বড় গলা কথাটা কার জন্য তৈরি হয়েছিল!", 1, 1, 1, 0, 0, 0, 0),
        ("আপনার চেহারা দেখলে মনে হয় স্বর্গের দূত, আর কাজ দেখলে নরকের কিটও লজ্জা পায়!", 1, 1, 1, 0, 0, 0, 0),
    ]

    for i, item in enumerate(sarcasm_implicit_sentences):
        text, tox, imp, sarc, prof, threat, ident, sex = item
        augmented_rows.append({
            "ID": f"aug_sarc_gen_{i}",
            "text": text,
            "clean_text": clean_bangla(text),
            "is_toxic": tox,
            "implicit_toxicity": imp,
            "sarcasm": sarc,
            "contains_profanity": prof,
            "threatening": threat,
            "identity_attack": ident,
            "sexual_content": sex,
            "is_augmented": 1
        })

    # =========================================================================
    # 3. Targeted Bengali Sexual Harassment / Vulgarity
    # =========================================================================
    sexual_sentences = [
        ("এই বেশ্যা মাগিটার শরীর দেখলে মাথা ঠিক থাকে না!", 1, 0, 0, 1, 0, 0, 1),
        ("তোর মতো নোংরা মেয়েদের রাস্তায় নামিয়ে বিবস্ত্র করা উচিত!", 1, 0, 0, 1, 1, 0, 1),
        ("শরীর দেখিয়ে লাইক পাওয়ার ধান্দা বন্ধ কর খঙ্কি মাগি!", 1, 0, 0, 1, 0, 0, 1),
        ("তোর মতো নটির কোনো চরিত্র নেই, বিছানায় গিয়ে টাকা কামাস!", 1, 0, 0, 1, 0, 0, 1),
        ("এমন শরীর নিয়ে বাইরে বের হলে পুরুষ তো জড়িয়ে ধরবেই!", 1, 1, 0, 0, 0, 0, 1),
        ("তোর স্তন আর নিতম্ব প্রদর্শন ছাড়া তো আর কোনো যোগ্যতা নেই!", 1, 0, 0, 1, 0, 0, 1),
        ("এই চরিত্রহীন মাগিটাকে দেখলেই থুতু মারতে ইচ্ছে করে!", 1, 0, 0, 1, 0, 0, 1),
        ("তোর রেট কত এক রাতের জন্য, ইনবক্সে বলিস তো!", 1, 0, 0, 0, 0, 0, 1),
        ("মেয়েরা ছোট কাপড় পরলে ধর্ষণ তো হবেই, আবার বিচার চায়!", 1, 0, 0, 0, 0, 1, 1),
        ("তোর মতো ছিনাল মেয়েদের সমাজ থেকে বয়কট করা দরকার!", 1, 0, 0, 1, 0, 0, 1),
    ]

    for i, item in enumerate(sexual_sentences):
        text, tox, imp, sarc, prof, threat, ident, sex = item
        augmented_rows.append({
            "ID": f"aug_sex_gen_{i}",
            "text": text,
            "clean_text": clean_bangla(text),
            "is_toxic": tox,
            "implicit_toxicity": imp,
            "sarcasm": sarc,
            "contains_profanity": prof,
            "threatening": threat,
            "identity_attack": ident,
            "sexual_content": sex,
            "is_augmented": 1
        })

    # =========================================================================
    # 4. True Benign Counterfactuals (Decoupling Identity Words from Toxicity)
    # =========================================================================
    benign_identity_sentences = [
        "হিন্দু মুসলিম বৌদ্ধ খ্রিস্টান আমরা সবাই বাংলাদেশের সমান নাগরিক।",
        "আমাদের সমাজে হিন্দু ও মুসলিম যুগ যুগ ধরে একসাথে সম্প্রীতির সাথে বাস করছে।",
        "ভারতের সাথে আমাদের বাণিজ্যিক ও সাংস্কৃতিক সম্পর্ক প্রাচীনকাল থেকেই বিদ্যমান।",
        "পাকিস্তানি তরুণ প্রজন্ম এবং আমাদের দেশের মানুষের মধ্যে শান্তি আলোচনা চলতে পারে।",
        "মেয়েরা আজ চিকিৎসা, বিজ্ঞান ও প্রশাসনে দেশের মুখ উজ্জ্বল করছে।",
        "ছেলেরা এবং মেয়েরা কাঁধে কাঁধ মিলিয়ে সুন্দর একটি দেশ গড়ে তুলছে।",
        "আওয়ামী লীগ এবং বিএনপি উভয় দলেরই উচিত জনগণের কল্যাণে গঠনমূলক রাজনীতি করা।",
        "বাঙালি এবং পাহাড়ি ক্ষুদ্র নৃগোষ্ঠীর মধ্যে পারস্পরিক শ্রদ্ধা বজায় রাখা প্রয়োজন।",
        "চাকমা মারমা ও বাঙালি সকলে মিলেই পার্বত্য চট্টগ্রামের সৌন্দর্য বৃদ্ধি করেছে।",
        "মসজিদ এবং মন্দিরের পবিত্রতা রক্ষা করা প্রতিটি বিবেকবান মানুষের দায়িত্ব।",
        "ধর্ম যার যার উৎসব সবার, এই ভ্রাতৃত্ববোধ আমাদের ধরে রাখতে হবে।",
        "মেয়েরা স্বাবলম্বী হলে পরিবারের পাশাপাশি দেশের সামগ্রিক অর্থনীতি শক্তিশালী হয়।",
        "আমাদের এলাকার হিন্দু ও মুসলমান প্রতিবেশীরা সবসময় সুখে-দুঃখে পাশে থাকে।",
        "ভারতের সাধারণ মানুষ এবং বাংলাদেশের সাধারণ মানুষ উভয়ই শান্তিপ্রিয়।",
        "রাজনীতিতে মতবিরোধ থাকবেই, কিন্তু সহিংসতা পরিহার করে সংলাপের পথ বেছে নিতে হবে।",
        "নারীর ক্ষমতায়ন ছাড়া কোনো জাতি কখনো অর্থনৈতিক মুক্তি অর্জন করতে পারে না।",
        "বাংলাদেশের সব ধর্মের মানুষের নিরাপত্তা ও অধিকার রক্ষা করা রাষ্ট্রের মূল নীতি।",
        "সকল রাজনৈতিক দলের লক্ষ্য হওয়া উচিত দারিদ্র্য দূরীকরণ ও কর্মসংস্থান সৃষ্টি।",
        "প্রত্যেক নাগরিকের নিজস্ব ধর্মীয় বিশ্বাস স্বাধীনভাবে পালন করার অধিকার আছে।",
        "পারস্পরিক ভালোবাসা ও শ্রদ্ধাই পারে সকল ভেদাভেদ দূর করে একটি সুন্দর সমাজ গড়তে।",
    ]

    for i, text in enumerate(benign_identity_sentences):
        augmented_rows.append({
            "ID": f"aug_benign_cf_{i}",
            "text": text,
            "clean_text": clean_bangla(text),
            "is_toxic": 0,
            "implicit_toxicity": 0,
            "sarcasm": 0,
            "contains_profanity": 0,
            "threatening": 0,
            "identity_attack": 0,
            "sexual_content": 0,
            "is_augmented": 1
        })

    # =========================================================================
    # 5. Identity-Swapped Benign Samples from Existing Benign Training Data
    # =========================================================================
    identity_pairs = [
        ("হিন্দু", "মুসলিম"),
        ("মুসলিম", "হিন্দু"),
        ("ভারত", "পাকিস্তান"),
        ("পাকিস্তান", "ভারত"),
        ("মেয়েরা", "ছেলেরা"),
        ("ছেলেরা", "মেয়েরা"),
        ("মন্দির", "মসজিদ"),
        ("মসজিদ", "মন্দির"),
        ("আওয়ামী লীগ", "বিএনপি"),
        ("বিএনপি", "আওয়ামী লীগ"),
        ("বাঙালি", "চাকমা"),
        ("চাকমা", "বাঙালি"),
        ("হিন্দুর", "মুসলিমের"),
        ("মুসলিমের", "হিন্দুর"),
        ("ভারতের", "পাকিস্তানের"),
        ("পাকিস্তানের", "ভারতের"),
    ]

    benign_orig = full_df[full_df[target_cols].sum(axis=1) == 0].copy()
    swap_count = 0
    for _, row in benign_orig.iterrows():
        text = row["clean_text"]
        for src, dst in sorted(identity_pairs, key=lambda pair: len(pair[0]), reverse=True):
            if src in text:
                swapped = text.replace(src, dst, 1)
                if swapped != text:
                    augmented_rows.append({
                        "ID": f"aug_swap_{swap_count}",
                        "text": swapped,
                        "clean_text": swapped,
                        "is_toxic": 0,
                        "implicit_toxicity": 0,
                        "sarcasm": 0,
                        "contains_profanity": 0,
                        "threatening": 0,
                        "identity_attack": 0,
                        "sexual_content": 0,
                        "is_augmented": 1
                    })
                    swap_count += 1
                break

    print(f"Generated {swap_count} identity-swapped benign samples.")

    # Combine original data and all augmented samples
    aug_df = pd.DataFrame(augmented_rows)
    combined_df = pd.concat([full_df, aug_df], ignore_index=True)
    combined_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n✅ Created {out_path} successfully!")
    print(f"Total rows: {len(combined_df)} (Original: {len(full_df)}, Augmented: {len(aug_df)})")
    print("\nAugmented class breakdown:")
    print(aug_df[target_cols].sum())
    print("\nCombined dataset class totals:")
    print(combined_df[target_cols].sum())

if __name__ == "__main__":
    main()
