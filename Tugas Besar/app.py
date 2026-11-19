from typing import TypedDict, List
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import time

#Langkah 1 : inisialisasi LLM Lokal (Ollama - qwen 2.5)
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

KNOWLEDGE_BASE = {
    "btree_join_where": "[KBP-001] Aturan Indexing: Kolom yang sering dipakai di klausul WHERE dan JOIN wajib dibuatkan B-Tree Index.",
    "composite_index": "[KBP-002] Aturan Composite Index: Jika ada pencarian gabungan dua kolom (misal WHERE status = 'X' AND date > 'Y'), buat composite index (status, date).",
    "order_by_index": "[KBP-003] Aturan Order By: Kolom pada klausul ORDER BY berukuran besar harus di-index untuk menghindari File Sort."
}

# Mensimulasikan hasil bacaan mesin database terkait bagian query mana yang bikin lambat (Full Table Scan)
def run_explain_plan_tool(sql_query: str) -> str:
    return (
        "EXPLAIN PLAN RESULT:\n"
        "- Node Type: Hash Join -> Seq Scan on transaksi t (cost=0.00..48210.50 rows=125000)\n"
        "- Filter Applied: lower(status_pembayaran) = 'lunas' AND date_part('year', tanggal_transaksi) = 2026\n"
        "- Sort Method: external merge Disk: 18420kB (ORDER BY t.total_harga DESC)"
    )

#Langkah 2 : Definisikan State
class AgentState(TypedDict, total=False):
    slow_query: str #berfungsi untuk input query sql yang lambat(slow query log)
    parser_analysis: str #output dari Query Parser Worker
    rag_recommendation: str #output dari Database Specialist Worker
    refactored_sql: str #output dari SQL Refactor Worker
    approval_status: str #Status persetujuan Human-in-the-Loop
    explain_plan: str #Menyimpan hasil bacaan EXPLAIN
    rag_status: str #Status pencarian Knowledge Base
    execution_result: str #Pesan sukses setelah DDL dijalankan
    next_worker: str #Nama Worker berikutnya yang ditentukan Supervisor

#Langkah 3 : Membuat Logika Worker

#Supervisor Node
def supervisor_node(state: AgentState):
    print("\n[SUPERVISOR] Memeriksa progres pengerjaan & menentukan langkah berikutnya...")

    if not state.get("parser_analysis"):
        print("[SUPERVISOR] (Routing) -> Query Parser Worker")
        return {"next_worker": "query_parser"}

    elif not state.get("rag_recommendation"):
        print("[SUPERVISOR] (Routing) -> Database Specialist Worker")
        return {"next_worker": "database_specialist"}

    elif not state.get("refactored_sql"):
        print("[SUPERVISOR] (Routing) -> SQL Refactor Worker")
        return {"next_worker": "sql_refactor"}

    elif not state.get("execution_result"):
        print("[SUPERVISOR] (Routing) -> Execute Index Node")
        return {"next_worker": "execute_index"}

    else:
        print("[SUPERVISOR] (Routing) -> Semua tahapan selesai! FINISH.")
        return {"next_worker": "FINISH"}

#WORKER 1 : Query Parser Worker
def query_parser_node(state: AgentState):
    print("\n(1)--- Query Parser Worker sedang menganalisis SQL.... ---")

    query = state["slow_query"]

    #ReAct Diagnostic Loop : Thought -> Action -> Observation
    thought = "Query memiliki JOIN, fungsi di WHERE, dan ORDER BY. Saya harus mengecek EXPLAIN plan terlebih dahulu."
    print(f"    [Thought (Pikiran Agen)] {thought}")

    print("    [Action (Tindakan)] Memanggil alat bantu run_explain_plan_tool()...")
    observation = run_explain_plan_tool(query)

    print(f"    [Observation (Hasil Baca)] {observation}")

    system_prompt = """Bertindaklah sebagai Database Performance Expert.
    Analisis Query berikut yang mengalami lambat (slow query) berdasarkan hasil EXPLAIN Plan.
    Identifikasilah bagian klausul mana yang menjadi Bottleneck (misal: WHERE, JOIN tanpa index, ORDER BY, dll).
    Berikan analisis singkat dan padat."""

    human_prompt = f"""Query SQL:{query} Berdasarkan hasil EXPLAIN Plan berikut:{observation} Identifikasilah bagian klausul mana yang menjadi Bottleneck dan jelaskan penyebabnya."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    return {
        "explain_plan": observation,
        "parser_analysis": response.content
    }

#Fungsi pencarian Knowledge Base
def search_knowledge_base(query: str):
    results = []
    query_lower = query.lower()

    keywords = [
        "where", "join", "index", "btree", "b-tree",
        "composite", "order", "sort", "status", "tanggal", "date"
    ]

    for key, knowledge in KNOWLEDGE_BASE.items():
        knowledge_lower = knowledge.lower()

        if any(word in query_lower and word in knowledge_lower for word in keywords):
            results.append(knowledge)

    return results

#WORKER 2 : Database Specialist Worker
def database_specialist_node(state: AgentState):
    print("\n(2)--- Database Specialist Worker sedang mencari aturan Indexing... ---")

    analysis = state["parser_analysis"]

    #RAG Search 1
    initial_query = "fix_lower_year_disk_sort_99"

    print(f"    [RAG Search 1] Mencari aturan dengan kata kunci awal: '{initial_query}'")

    first_results = search_knowledge_base(initial_query)

    if not first_results:
        status = "NOT_FOUND"

        print(f"    [Status RAG] {status} -> Gagal! Melakukan Self-Correction (Query Rewriting)...")

        rewrite_system_prompt = """Kamu adalah Database Indexing Expert.
        Ubah analisis masalah SQL menjadi kata kunci pencarian Knowledge Base.
        Fokus pada konsep B-Tree Index, WHERE, JOIN, Composite Index, ORDER BY, dan fungsi pada kolom.
        Jawab hanya kata kunci pencarian dalam 1 baris."""

        rewrite_human_prompt = f"""Analisis masalah SQL berikut: {analysis}"""

        rewritten_query = llm.invoke([
            SystemMessage(content=rewrite_system_prompt),
            HumanMessage(content=rewrite_human_prompt)
        ]).content.strip()

        print(f"    [Query Rewriting] Berhasil menyusun kata kunci baru: '{rewritten_query}'")

        #RAG Search 2 setelah Self-Correction
        print("    [RAG Search 2] Mencari kembali Knowledge Base menggunakan query hasil rewriting...")

        second_results = search_knowledge_base(rewritten_query)

        if second_results:
            status = "FOUND"
            knowledge_base = "\n".join(second_results)

            print(f"    [Status RAG] {status}! Aturan Knowledge Base berhasil ditarik.")

        else:
            status = "NOT_FOUND"
            knowledge_base = "Tidak ditemukan aturan yang relevan."

            print(f"    [Status RAG] {status}")

    else:
        status = "FOUND"
        knowledge_base = "\n".join(first_results)

        print(f"    [Status RAG] {status}! Aturan Knowledge Base berhasil ditarik.")

    prompt = f"""Bertindaklah sebagai Database Indexing Specialist.
    Berdasarkan hasil analisis query berikut:{analysis} 
    Dan menggunakan panduan pada Knowledge Base:
    {knowledge_base}
    Tentukan rekomendasi jenis index apa yang paling tepat untuk memperbaiki performa query tersebut.
    Jelaskan kolom yang perlu di-index, jenis index, apakah membutuhkan Composite Index, dan alasannya."""

    response = llm.invoke(prompt)

    return {
        "rag_status": status,
        "rag_recommendation": response.content
    }

#WORKER 3 : SQL Refactor Worker
def sql_refactor_node(state: AgentState):
    print("\n(3)--- SQL Refactor Worker menyusun query SQL efisien dan DDL Index... ---")

    query = state["slow_query"]
    recommendation = state["rag_recommendation"]

    system_prompt = """Bertindaklah sebagai Senior SQL Performance Engineer.

    Aturan Mutlak Refactoring SQL:
    1. DILARANG KERAS menulis SELECT * pada query hasil optimasi. Wajib menggunakan daftar kolom eksplisit.
    2. DILARANG KERAS menggunakan fungsi seperti LOWER(), UPPER(), YEAR(), DATE(), DATE_PART() pada kolom di WHERE atau JOIN agar tetap SARGable.
    3. Jika filtering berdasarkan tahun diperlukan, gunakan date range.
    Contoh: tanggal_transaksi >= '2026-01-01' AND tanggal_transaksi < '2027-01-01'.
    4. Susun perintah DDL CREATE INDEX berdasarkan rekomendasi Database Specialist.
    5. Jangan mengarang nama kolom yang tidak terlihat pada query.
    6. Kolom yang terlihat dari query adalah:
    t.id_pelanggan, t.status_pembayaran, t.tanggal_transaksi,
    t.total_harga, dan p.id_pelanggan.

    Jawab HANYA dengan format berikut:

    ### 1. Query SQL Optimal
    ```sql
    (tulis query hasil refactor di sini)

    (tulis perintah CREATE INDEX di sini)

    (jelaskan perubahan yang dilakukan secara singkat)"""

    human_prompt = f"""Query Asal: {query} Rekomendasi Indexing:{recommendation}"""
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    return {"refactored_sql": response.content}

    #WORKER 4 : Execute Index Node
def execute_index_node(state: AgentState):
    print("\n(4)--- Execute Index Node mengeksekusi DDL ke DB Server... ---")
    time.sleep(1)

    #Pada implementasi nyata, bagian ini dapat dihubungkan ke PostgreSQL/MySQL.
    #Untuk project ini, eksekusi DDL disimulasikan.

    hasil = "DDL CREATE INDEX berhasil dieksekusi ke DB Server. Index aktif & statistik tabel diperbarui."

    print(f"    [DB SERVER] {hasil}")

    return {"execution_result": hasil}

#Langkah 4 : Merakit Grafik dan HUMAN-IN-THE-LOOP (Interrupt)

#MemorySaver untuk menyimpan state saat graf dijeda
memory = MemorySaver()

#Buat StateGraph berdasarkan AgentState
workflow = StateGraph(AgentState)

#Daftarkan semua Node (Worker)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("query_parser", query_parser_node)
workflow.add_node("database_specialist", database_specialist_node)
workflow.add_node("sql_refactor", sql_refactor_node)
workflow.add_node("execute_index", execute_index_node)

#Tentukan Alur (Edges)
workflow.add_edge(START, "supervisor")

#Supervisor menentukan Worker berdasarkan isi State
workflow.add_conditional_edges(
"supervisor",
lambda state: state["next_worker"],
{
"query_parser": "query_parser",
"database_specialist": "database_specialist",
"sql_refactor": "sql_refactor",
"execute_index": "execute_index",
"FINISH": END
}
)

#Setelah Worker selesai, kembali ke Supervisor
workflow.add_edge("query_parser", "supervisor")
workflow.add_edge("database_specialist", "supervisor")
workflow.add_edge("sql_refactor", "supervisor")
workflow.add_edge("execute_index", "supervisor")

#Compile GRAF dengan fitur interrupt_before
#Graf akan BERHENTI OTOMATIS sebelum execute_index
#agar manusia/DBA dapat mengecek hasil DDL terlebih dahulu.
app = workflow.compile(checkpointer=memory,interrupt_before=["execute_index"])
print("LangGraph Berhasil diCompile!")

#Langkah 5 : TESTING
if __name__ == "__main__":

    #Karena menggunakan Human-in-the-Loop dan MemorySaver()
    #maka sistem membutuhkan Thread ID.
    config = {"configurable": {"thread_id": "tubes-ai-01"}}

    initial_input = {
        "slow_query": """
SELECT *
FROM transaksi t
JOIN pelanggan p
ON t.id_pelanggan = p.id_pelanggan
WHERE LOWER(t.status_pembayaran) = 'lunas'
  AND YEAR(t.tanggal_transaksi) = 2026
ORDER BY t.total_harga DESC;
"""
    }

    print("Memulai eksekusi Multi-Agent System...\n")

    #Jalankan program sampai menyentuh titik jeda (interrupt)
    for event in app.stream(initial_input, config):
        print(f"Event Terdeteksi : {list(event.keys())}")

    #Mengambil kondisi memori terakhir saat program dijeda
    current_state = app.get_state(config)

    print("\nSistem Berhenti untuk Human Review (HITL)!")

    print("\n--- EXPLAIN PLAN ---")
    print(current_state.values.get("explain_plan"))

    print("\nHasil Analisis Parser:")
    print(current_state.values.get("parser_analysis"))

    print("\nStatus RAG:")
    print(current_state.values.get("rag_status"))

    print("\nRekomendasi RAG:")
    print(current_state.values.get("rag_recommendation"))

    print("\n--- HASIL REFACTOR SQL & RANCANGAN DDL ---")
    print(current_state.values.get("refactored_sql"))

    persetujuan = input(
        "\nApakah Anda menyetujui eksekusi DDL Index ini ke DB Server (y/n): "
    )

    if persetujuan.lower() == "y":
        print("\nDisetujui! Melanjutkan Eksekusi ke Execute Index Node....")

        #Menyimpan status approval ke State
        app.update_state(
            config,
            {"approval_status": "APPROVED"}
        )

        #Melanjutkan alur program dari titik berhenti
        for event in app.stream(None, config):
            print(f"Event Terdeteksi : {list(event.keys())}")

        final_state = app.get_state(config)

        print("\n--- HASIL AKHIR EKSEKUSI DATABASE ---")
        print(final_state.values.get("execution_result"))
        print("Approval Status:", final_state.values.get("approval_status"))

    else:
        print("\nPerintah dibatalkan oleh user.")

        app.update_state(
            config,
            {"approval_status": "REJECTED"}
        )

        print("Approval Status: REJECTED")