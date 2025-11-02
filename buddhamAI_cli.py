# buddhamAI_cli.py
import sys
import os
import pickle
import json
import time
import subprocess
import traceback
import numpy as np
import faiss
import ollama
import hashlib
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from reDocuments import ensure_embeddings_up_to_date
from debugger import format_duration, log

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    
    def clear_screen():
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/Mac
            os.system('clear')
    
    log_file = "buddhamAI_cli.log"
    required_models = ["gpt-oss:20b", "nomic-embed-text:v1.5"]
    EMB_PATH = "embeddings.npy"
    META_PATH = "metadata.pkl"
    DOCS_ALL_PATH = "documents/documentsPkl/documents_all.pkl"
    start = None
    end = None
    STATUS_FILE = "embed_status.json"
    debug_mode = os.getenv("DEBUG", "false").lower()
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("")

    def get_installed_models():
        # try --json
        try:
            result = subprocess.run(
            ["ollama", "list", "--json"],
            capture_output=True,
            text=True
            )
            if result.returncode == 0 and result.stdout.strip().startswith("["):
                return [m["name"] for m in json.loads(result.stdout)]
        except Exception:
            pass
        # fallback to parsing text output
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split("\n")
        models = []
        for line in lines[1:]:  # skip header
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models

    def check_and_pull_models(models_to_check):
        try:
            local_model_names = get_installed_models()
            missing_models = [m for m in models_to_check if m not in local_model_names]

            if missing_models:
                for model in missing_models:
                    log(f"📥 Installing {model} ...")
                    subprocess.run(["ollama", "pull", model], check=True)
                    log(f"✅ Finished installing {model}")
            else:
                log("✅ All models are present")
        except Exception:
            log("❌ Error:\n" + traceback.format_exc())
            exit(1)

    def flatten_docs(raw):
        docs = []
        for book_name, chapters in raw.items():
            for chapter_key, pages in chapters.items():
                for page_key, content in pages.items():
                    chapter_num = int(chapter_key.replace("chapter ", ""))
                    docs.append({
                        "bookname": book_name,
                        "chapter": chapter_num,
                        "content": content
                    })
        return docs

    def load_embeddings_and_metadata():
        log(f"use embeddings {required_models[1]}")
        log(f"Loading {EMB_PATH} and {META_PATH}")
        embeddings = np.load(EMB_PATH)
        with open(META_PATH, 'rb') as f:
            metadata = pickle.load(f)
        return embeddings, metadata

    def search(query, index, metadata, top_k, max_distance):
        log(f"Searching for references for: {query} with top_k={top_k} and max_distance={max_distance}")

        # Create embedding for query
        q_emb = np.array([ollama.embeddings(model='nomic-embed-text:v1.5', prompt=query)['embedding']], dtype='float32')

        # Fetch more from FAISS เผื่อ filter max_distance
        fetch_k = top_k * 5
        distances, ids = index.search(q_emb, fetch_k)
        log(f"Found {len(ids[0])} nearest neighbors (fetched {fetch_k})")

        results = []
        filtered_out_results = []
        seen_docs = set()

        for i, idx in enumerate(ids[0]):
            if idx >= len(metadata):
                continue

            dist = distances[0][i]
            doc = metadata[idx]
            doc_hash = hashlib.md5(doc['content'].encode('utf-8')).hexdigest()

            # filter duplicates และ max_distance
            if doc_hash in seen_docs or (max_distance is not None and dist > max_distance):
                filtered_out_results.append((doc, dist, idx))
                log(f"index={idx}, distance={dist:.4f} removed")
                continue

            results.append({
                "doc": doc,
                "distance": dist,
                "index": idx
            })
            seen_docs.add(doc_hash)
            log(f"index={idx}, distance={dist:.4f} added")

            # stop เมื่อได้ top_k
            if len(results) >= top_k:
                break

        log(f"Searching for references found {len(results)} items: {short_references([r['doc'] for r in results])}")

        if filtered_out_results:
            contexts = [f"{doc['content']}" for doc, _, _ in filtered_out_results]
            full_context = "\n".join(contexts)
            log(f"Filtered out references {len(filtered_out_results)} items: {short_references([doc for doc, _, _ in filtered_out_results])}")
            log(f"Filtered out contexts:\n{full_context}")

        return results

        
    def short_references(metadata):
        sorted_docs = sorted(metadata, key=lambda d: (d['bookName'], d['chapterName']))
        return ", ".join([
            f"{d['bookName']} - {d['chapterName']}"
            for d in sorted_docs
        ])

    def filter_user_query(query: str):
        # ตรวจสอบว่า EMB_PATH โหลดได้ไหม
        try:
            emb_matrix = np.load(EMB_PATH)
        except Exception as e:
            log(f"Error loading EMB_PATH: {EMB_PATH}, exception: {e}")
            return None

        log(f"Loaded emb_matrix shape: {emb_matrix.shape}")
        
        # ดึง embedding ของ query
        query_emb = np.array(
            ollama.embeddings(model="nomic-embed-text:v1.5", prompt=query)["embedding"]
        )
        log(f"query_emb shape: {query_emb.shape}")

        # ตรวจสอบ norm ของ query
        query_norm = np.linalg.norm(query_emb)
        log(f"query_emb norm: {query_norm:.6f}")
        
        # ตรวจสอบ norm ของทุก embedding
        emb_norms = np.linalg.norm(emb_matrix, axis=1)
        zero_indices = np.where(emb_norms == 0)[0]
        if query_norm == 0:
            log("Warning: query embedding is zero vector!")
        if len(zero_indices) > 0:
            log(f"Warning: found zero vector(s) in embeddings at indices {zero_indices}")
        
        # คำนวณ cosine similarity ด้วย sklearn
        similarities = cosine_similarity(query_emb.reshape(1, -1), emb_matrix)[0]
        
        top_index = np.argmax(similarities)
        log(f"ค้นหาด้วย '{query}' เจอข้อมูลใกล้สุดที่ index {top_index}, similarity = {similarities[top_index]:.6f}")
        
        return similarities[top_index]

    def check_rejection_message(text: str) -> bool:
        rejection_phrases = [
            "ไม่สามารถตอบคำถาม",
            "ไม่แน่ใจว่าต้องการขออะไรจากข้อมูลนี้",
            "คุณช่วยชี้แจงเพิ่มเติมได้ไหม",
            "กรุณาแจ้งชัดเจน",
            "เพื่อให้ตอบได้ตรงประเด็น",
            "ไม่สามารถตอบ",
            "ไม่สามารถบอกได้",
            "ไม่เข้าใจว่าคุณต้องการถามอะไร",
            "ระบุคำถามให้ชัดเจน"
        ]
        return any(phrase in text for phrase in rejection_phrases)
    
    def check_greeting_message(text: str) -> bool:
        greeting_phrases = [
            "สวัสดีครับ",
        ]
        return any(phrase in text for phrase in greeting_phrases)

    def ask(query, index, metadata, top_k=None, max_distance=None):
        global start
        start = time.perf_counter()
        top_k = len(query) if top_k is None else top_k
        if top_k > 10:
            top_k = 10

        if filter_user_query(query) < 0.4:
            end = time.perf_counter()
            processing_time = format_duration(end - start)
            return {
                "answer": "ขออภัยครับ...ผมไม่สามารถตอบคำถามนี้ได้....เนื่องจากผมถูกออกแบบมาเพื่อให้ตอบคำถามเกี่ยวกับพระพุทธธรรมเท่านั้น",
                "duration": f'ใช้เวลา {processing_time}'
            }
        else :
            results = search(query, index, metadata, top_k, max_distance)

        contexts = [r["doc"]["content"] for r in results]
        
        full_context = "\n".join(contexts)
        prompt = f"""ข้อมูลอ้างอิง:\n{full_context}\nคำถาม: {query}"""
        model = 'gpt-oss:20b'
        log(f"Asking model: \"{model}\" with prompt:\n{prompt}")
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "คุณคือผู้ช่วยที่ใช้ข้อมูลอ้างอิงจากเอกสารหลายแหล่ง ซึ่งจะช่วยตอบคำถามเกี่ยวกับพระพุทธรรมและจะตอบด้วยภาษาไทยเท่านั้นโดยเช็คจากคำถามก่อนซึ่งจะระบุในบรรทัดสุดท้ายถ้าได้รับคำถามนอกเหนือจากพระพุทธรรมให้ตอบว่า ขออภัยครับ...ผมไม่สามารถตอบคำถามนี้ได้....เนื่องจากผมถูกออกแบบมาเพื่อให้ตอบคำถามเกี่ยวกับพระพุทธธรรมเท่านั้น"},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response['message']['content']
        if filter_user_query(answer) < 0.7:
            end = time.perf_counter()
            processing_time = format_duration(end - start)
            return {
                "answer": "ขออภัยครับ...ผมไม่สามารถตอบคำถามนี้ได้....เนื่องจากผมถูกออกแบบมาเพื่อให้ตอบคำถามเกี่ยวกับพระพุทธธรรมเท่านั้น",
                "duration": f'ใช้เวลา {processing_time}'
            }
        ref_text = short_references([r["doc"] for r in results])
        end = time.perf_counter()
        processing_time = format_duration(end - start)
        log(f"Asked \"{model}\" finished in {processing_time}")
        if check_rejection_message(answer) or check_greeting_message(answer):
            return {
                "answer": answer,
                "duration": f'ใช้เวลา {processing_time}'
            }
        else:
            return {
                "answer": answer,
                "references": f"อ้างอิงข้อมูลจาก\n {ref_text}",
                "duration": f'ใช้เวลา {processing_time}'
            }
    
    def read_last_embed_time():
        if not os.path.exists(STATUS_FILE):
            return "1970-01-01 00:00:00"
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)["last_embed_time"]

    def init_bot():
        log("Checking for data updates...")
        embeddings, metadata = ensure_embeddings_up_to_date()
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, metadata

    def parse_args(argv):
        message = None
        top_k = None
        max_distance = None

        i = 0
        while i < len(argv):
            arg = argv[i]
            if arg == '-k' and i + 1 < len(argv):
                try:
                    top_k = int(argv[i+1])
                except ValueError:
                    top_k = None
                i += 2
            elif arg == '-d' and i + 1 < len(argv):
                try:
                    max_distance = float(argv[i+1])
                except ValueError:
                    max_distance = None
                i += 2
            elif not arg.startswith('-') and message is None:
                message = arg
                i += 1
            else:
                i += 1

        return message, top_k, max_distance


    def ask_cli(argv=None):
        try:
            log(f"Starting BuddhamAI with argv: {argv}")
            if debug_mode == "false":
                check_and_pull_models(required_models)

            if argv is None:  # if not provided → use sys.argv
                argv = sys.argv[1:]
                message, top_k, max_distance = parse_args(argv)
                
            log(f"Parsed arguments - message: {message}, top_k: {top_k}, max_distance: {max_distance}")

            if message is None or message.strip() == "":
                    result = {"answer": "กรุณาถามคำถาม", "references": "ไม่มี", "duration": format_duration(0)}
                    data = {"data": result}
                    json_str = json.dumps(data, ensure_ascii=False)
                    log(json_str)
                    print(json_str)
                    return data  # return to main.py

            index, metadata = init_bot()
            if debug_mode == "true":
                time.sleep(float(os.getenv("DEBUG_TIME")))
                result = {"answer": "**ธรรม (Dhamma) คือ สิ่งที่พระพุทธเจ้าสอนเพื่อพ้นทุกข์**  - **ความหมายโดยรวม**: “ธรรม” หมายถึง กฎและหลักธรรมที่พระพุทธเจ้าผูกให้สอนว่าเป็นเส้นทางสู่ความเข้าใจจริงของชีวิตและการดับทุกข์  - **ความสัมพันธ์กับอริยสัจ 4**:    1. **ทุกข์** – ความทุกข์และความไม่สบายใจในชีวิต    2. **สมุทัย** – เหตุของทุกข์ (ความตัณหา, ความอยาก)    3. **นิโรธ** – การดับทุกข์ (สันติความสุขที่ไม่มีความทุกข์)    4. **มรรค** – เส้นทาง (อริยอุปสรรค) ที่นำไปสู่การดับทุกข์  - **วิธีการดำเนินชีวิตตามธรรม**:    * ฝึกสติและปฏิบัติในปัจจุบัน    * เรียนรู้และทำตามมรรค (เส้นทาง 8 ส่วน)    * ลดตัณหาและแสวงหาปัญญาจากความจริงของธรรม  สรุปได้ว่า “ธรรม” คือหลักการและกฎที่สอนให้มนุษย์เข้าใจความเป็นจริงของชีวิตและปกป้องตนเองจากความทุกข์ โดยการปฏิบัติตามอริยสัจ 4 และมรรค.", "argv": message, "references": "test only", "duration": format_duration(0)} # for test only
                data = {"data": result}
                json_str = json.dumps(data, ensure_ascii=False)
                log(json_str)
                print(json_str)
                return data  # return to main.py
            
            result = ask(message, index, metadata, top_k=top_k, max_distance=max_distance)

            data = {"data": result}
            json_str = json.dumps(data, ensure_ascii=False)
            log(json_str)
            print(json_str)
            return data  # return to main.py
        except Exception:
            err_msg = traceback.format_exc()
            log("Error: " + err_msg)
            result = {"Error": f"Error: {err_msg}", "status": 500}
            data = {"data": result}
            json_str = json.dumps(data, ensure_ascii=False)
            log(json_str)
            print(json_str)
            return data  # return to main.py

    if __name__ == "__main__":
        ask_cli()
        
except Exception:
    err_msg = traceback.format_exc()
    try:
        log("Error: " + err_msg)
        result = {"Error": f"Error: {err_msg}", "status": 500}
        data = {"data": result}
        json_str = json.dumps(data, ensure_ascii=False)
        log(json_str)
        print(json_str)
    except:
        log("Error: " + err_msg)
        result = {"Error": f"Error: {err_msg}", "status": 500}
        data = {"data": result}
        json_str = json.dumps(data, ensure_ascii=False)
        log(json_str)
        print(json_str)
    sys.exit(1)