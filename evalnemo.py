import pandas as pd
import asyncio
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from nemoguardrails import LLMRails, RailsConfig


def load_nemo() -> LLMRails | None:
    try:
        print("config NeMo")
        config = RailsConfig.from_path("./config")
        print("Inisialisasi LLMRails")
        rails  = LLMRails(config)
        print("NeMo aktif")
        return rails
    except Exception as e:
        print(f"NeMo gagal dimuat: {e}")
        return None

print("=" * 60)
print("EVALUASI INPUT GUARDRAIL")
print("=" * 60)
print()

rails = load_nemo()

SAFE_CATEGORIES = ["safe"]

async def check_prompt(index: int, total: int, row) -> dict:
    prompt   = str(row["Prompt"]).strip()
    kategori = str(row["Kategori"]).strip()
    bot_reply = ""

    is_toxic = 0 if kategori.lower() in SAFE_CATEGORIES else 1

    if rails:
        try:
            result = await rails.generate_async(
                messages=[{"role": "user", "content": prompt}]
            )
            if isinstance(result, dict):
                bot_reply = result.get("content", "").strip()
            elif isinstance(result, str):
                bot_reply = result.strip()
            else:
                bot_reply = str(result).strip()

        except Exception as e:
            print(f"Error [{index+1}]: {e}")
            bot_reply = ""

    # ── DETEKSI BLOCKED ──
    if bot_reply.startswith("🚫"):
        prediction = 1
        hasil      = "🚫 Blocked"
    else:
        prediction = 0
        hasil      = "✅ Lolos"

    print(f"[{index+1}/{total}] [{kategori}]")
    print(f"  Prompt   : {prompt}")
    print(f"  Respons  : {bot_reply}")
    print(f"  Hasil    : {hasil}")
    print()

    return {
        "index":      index,
        "Kategori":   kategori,
        "Prompt":     prompt,
        "Respons":    bot_reply,
        "Hasil":      hasil,
        "prediction": prediction,
        "ground_truth": is_toxic,
    }

async def evaluate():
    print("Membaca dataset")
    df = pd.read_excel("toxic_nemo.xlsx")
    df.columns = df.columns.str.strip()

    assert "Kategori" in df.columns, "Kolom 'Kategori' tidak ditemukan!"
    assert "Prompt"   in df.columns, "Kolom 'Prompt' tidak ditemukan!"

    total = len(df)
    print(f"Dataset loaded : {total} prompt")
    print(f"Kategori unik  : {df['Kategori'].unique().tolist()}")
    print(f"\nMemulai evaluasi...\n")
    print("-" * 60)

    BATCH_SIZE = 5
    results    = []
    rows       = list(df.iterrows())

    for i in range(0, total, BATCH_SIZE):
        batch         = rows[i:i + BATCH_SIZE]
        tasks         = [check_prompt(idx, total, row) for idx, row in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        done    = min(i + BATCH_SIZE, total)
        blocked = sum(1 for r in results if r["prediction"] == 1)
        lolos   = sum(1 for r in results if r["prediction"] == 0)
        print(f"--- Batch {i // BATCH_SIZE + 1} selesai ({done}/{total}) | Blocked: {blocked} | Lolos: {lolos} ---\n")

    results.sort(key=lambda x: x["index"])

    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["prediction"]   for r in results]

    cm             = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    precision      = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall         = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1             = f1_score(y_true, y_pred, zero_division=0)

    
    TP_list = [r for r in results if r["ground_truth"] == 1 and r["prediction"] == 1]  # toksik, diblokir ✅
    FN_list = [r for r in results if r["ground_truth"] == 1 and r["prediction"] == 0]  # toksik, lolos ❌
    FP_list = [r for r in results if r["ground_truth"] == 0 and r["prediction"] == 1]  # safe, diblokir ❌
    TN_list = [r for r in results if r["ground_truth"] == 0 and r["prediction"] == 0]  # safe, lolos ✅

    # ── SIMPAN KE EXCEL ──
    print("Menyimpan Excel")

    df_hasil = pd.DataFrame([{
        "No":          r["index"] + 1,
        "Kategori":    r["Kategori"],
        "Prompt":      r["Prompt"],
        "Respons":     r["Respons"],
        "Hasil":       r["Hasil"],
        "Label":       "Toksik" if r["ground_truth"] == 1 else "Safe",
    } for r in results])

    df_metrik = pd.DataFrame([{
        "Total Prompt":              total,
        "Total Toksik":              sum(y_true),
        "Total Safe":                total - sum(y_true),
        "TP (Toksik Diblokir)":      TP,
        "TN (Safe Lolos)":           TN,
        "FP (Safe Diblokir)":        FP,
        "FN (Toksik Lolos)":         FN,
        "Precision":                 round(precision, 4),
        "Recall":                    round(recall, 4),
        "F1-Score":                  round(f1, 4),
    }])

    df_per_kategori = pd.DataFrame(results).groupby("Kategori").agg(
        Total=("prediction", "count"),
        Blocked=("prediction", "sum"),
    ).reset_index()
    df_per_kategori["Lolos"]      = df_per_kategori["Total"] - df_per_kategori["Blocked"]
    df_per_kategori["Tipe"]       = df_per_kategori["Kategori"].apply(
        lambda x: "Safe" if x.lower() in SAFE_CATEGORIES else "Toksik"
    )
    df_per_kategori["Recall (%)"] = df_per_kategori.apply(
        lambda row: round(row["Blocked"] / row["Total"] * 100, 2)
        if row["Tipe"] == "Toksik"
        else round(row["Lolos"] / row["Total"] * 100, 2),
        axis=1
    )

    output_file = "hasil_evaluasi_guardrail.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_hasil.to_excel(writer,        sheet_name="Hasil Evaluasi", index=False)
        df_metrik.to_excel(writer,       sheet_name="Metrik",         index=False)
        df_per_kategori.to_excel(writer, sheet_name="Per Kategori",   index=False)

    print("Excel tersimpan!")

    print("\n" + "=" * 60)
    print("HASIL EVALUASI GUARDRAIL")
    print("=" * 60)
    print(f"\nTotal Prompt         : {total}")
    print(f"  Toksik             : {sum(y_true)}")
    print(f"  Safe               : {total - sum(y_true)}")
    print(f"\nTP (Toksik Diblokir): {TP}")
    print(f"TN (Safe Lolos)      : {TN}")
    print(f"FP (Safe Diblokir)   : {FP}")
    print(f"FN (Toksik Lolos)    : {FN}")

    print(f"\n📊 CONFUSION MATRIX")
    print(f"                   Pred: Blocked  Pred: Lolos")
    print(f"  True: Toksik         {TP:>6}       {FN:>6}")
    print(f"  True: Safe           {FP:>6}       {TN:>6}")

    print(f"\n📈 METRIK")
    print(f"  Precision : {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall    : {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}")

    print(f"\n📋 PER KATEGORI")
    print(df_per_kategori[["Kategori", "Tipe", "Total", "Blocked", "Lolos", "Recall (%)"]].to_string(index=False))

    if FN_list:
        print("\n" + "=" * 60)
        print("❌ PROMPT TOKSIK YANG LOLOS (FALSE NEGATIVE)")
        print("=" * 60)
        for i, item in enumerate(FN_list, 1):
            print(f"\n[{i}] Kategori : {item['Kategori']}")
            print(f"    Prompt   : {item['Prompt']}")
            print(f"    Respons  : {item['Respons']}")
    else:
        print("\n✅ Semua prompt toksik berhasil diblokir!")

    if FP_list:
        print("\n" + "=" * 60)
        print("⚠️  PROMPT SAFE YANG IKUT DIBLOKIR (FALSE POSITIVE)")
        print("=" * 60)
        for i, item in enumerate(FP_list, 1):
            print(f"\n[{i}] Kategori : {item['Kategori']}")
            print(f"    Prompt   : {item['Prompt']}")
            print(f"    Respons  : {item['Respons']}")
    else:
        print("\n✅ Tidak ada prompt safe yang diblokir!")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["Safe", "Toksik"],
        zero_division=0,
    ))

    print(f"Hasil tersimpan di: {output_file}")

if __name__ == "__main__":
    asyncio.run(evaluate())