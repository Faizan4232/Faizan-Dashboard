import os

os.makedirs("data", exist_ok=True)

test_path = "data/_test_write.txt"
try:
    with open(test_path, "w") as f:
        f.write("ok")
    print("✅ Write OK:", test_path)
except Exception as e:
    print("❌ Write FAILED:", e)
