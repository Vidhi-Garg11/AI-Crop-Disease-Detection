import os
from gradio_client import Client, handle_file

c = Client("http://127.0.0.1:7860", verbose=False)
base = r"C:\Users\PG Sravani\Downloads\IVA CIA-3\dataset\valid"
classes = sorted(os.listdir(base))
print(f"{'TRUE CLASS':38s} -> {'TOP PREDICTION':38s} CORRECT")
print("-" * 90)
ok = 0
total = 0
for cls in classes:
    files = sorted(os.listdir(os.path.join(base, cls)))
    for fn in (files[0], files[len(files) // 2]):
        img = os.path.join(base, cls, fn)
        res = c.predict(handle_file(img))
        label = res["label"]
        is_correct = label == cls
        ok += is_correct
        total += 1
        print(f"{cls:38s} -> {label:38s} {is_correct}")
print("-" * 90)
print(f"Correct: {ok}/{total} = {ok / total * 100:.0f}%")