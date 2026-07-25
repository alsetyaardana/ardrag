# Ardrag — Langkah Uji Coba

Panduan ini untuk verifikasi manual bahwa stack (Qdrant + backend API + web UI + MCP server)
berjalan benar, baik di lokal (Docker Desktop) maupun setelah deploy ke VPS.

## 0. Prasyarat

- Docker & Docker Compose terpasang.
- Untuk uji lokal: buat network eksternal dulu (biasanya sudah ada di VPS lewat setup Cloudflare Tunnel):
  ```bash
  docker network create proxy-net
  ```
- Salin `.env.example` ke `.env`.

## 1. Build & jalankan stack

```bash
docker compose up -d --build
docker compose ps
```

**Ekspektasi:** dua container `qdrant` dan `ardrag-backend` berstatus `Up`.

```bash
docker compose logs ardrag-backend --tail 30
```

**Ekspektasi:** ada baris `Uvicorn running on http://0.0.0.0:8000` dan
`Starting MCP server 'Ardrag' with transport 'sse' on http://0.0.0.0:8001/sse`,
tanpa traceback error.

## 2. Cek resource usage

```bash
docker stats --no-stream
```

**Ekspektasi (skala dokumen kecil):** total memory idle di kisaran 600-900MB, CPU mendekati 0%
saat idle. Kalau ini jauh melebihi alokasi VPS kamu, itu tanda perlu investigasi lebih lanjut
sebelum lanjut ke produksi.

## 3. Cek Web UI & API dasar

```bash
curl -s -o /dev/null -w "UI status: %{http_code}\n" http://localhost:8000/
curl -s http://localhost:8000/documents
```

**Ekspektasi:** UI status `200`, daftar dokumen `[]` (kosong di awal).

Buka `http://localhost:8000` di browser — pastikan halaman "Ardrag — Document Manager" tampil
dengan teks terbaca jelas (baik light maupun dark mode browser).

## 4. Uji upload dokumen baru

```bash
echo "Ardrag adalah RAG pribadi untuk MCP." > test.txt
curl -s -F "file=@test.txt" http://localhost:8000/documents
```

**Ekspektasi:** respons `"status":"created"` dengan `chunk_count` > 0.

Cek juga lewat UI: refresh halaman, dokumen `test.txt` harus muncul di tabel dengan jumlah chunk
dan waktu upload yang sesuai.

## 5. Uji pencarian (retrieval)

```bash
curl -s "http://localhost:8000/search?q=RAG%20pribadi"
```

**Ekspektasi:** hasil array berisi chunk dari `test.txt` dengan `score` yang masuk akal (>0.5
untuk query yang relevan).

## 6. Uji replace dokumen (upload ulang nama sama)

```bash
echo "Konten baru: topik yang sama sekali berbeda, tentang kucing." > test.txt
curl -s -F "file=@test.txt" http://localhost:8000/documents
curl -s "http://localhost:8000/search?q=kucing"
curl -s "http://localhost:8000/search?q=RAG%20pribadi"
```

**Ekspektasi:**
- Respons upload: `"status":"replaced"`, `id` dokumen **tetap sama** dengan langkah 4.
- Search "kucing" → mengembalikan chunk baru.
- Search "RAG pribadi" (konten lama) → **tidak lagi muncul** (skor rendah/kosong), membuktikan
  chunk lama sudah terhapus, bukan menumpuk.

## 6b. Uji batch upload (banyak file sekaligus)

```bash
echo "Dokumen satu: tentang kucing." > a.txt
echo "Dokumen dua: tentang anjing." > b.txt
echo "Dokumen tiga: tentang burung." > c.txt
curl -s -F "files=@a.txt" -F "files=@b.txt" -F "files=@c.txt" http://localhost:8000/documents/batch
```

**Ekspektasi:** respons `"total":3,"succeeded":3,"failed":0` dengan detail per file di `results`.

Cek juga lewat UI: klik "Choose Files", pilih beberapa file sekaligus (bukan satu-satu), klik
Upload — status ringkas "X/Y succeeded" muncul, dan semua dokumen tampil di tabel.

**Uji partial failure:** sisipkan satu file kosong/tidak valid di antara file yang valid, pastikan
file yang valid tetap ter-index (`succeeded` > 0) dan file yang gagal muncul di `results` dengan
`status: "error"` beserta alasannya — tidak menggagalkan seluruh batch.

## 7. Uji hapus dokumen

```bash
curl -s http://localhost:8000/documents
# catat "id" dari test.txt, misal 1
curl -s -X DELETE http://localhost:8000/documents/1
curl -s http://localhost:8000/documents
curl -s "http://localhost:8000/search?q=kucing"
```

**Ekspektasi:** setelah delete, `GET /documents` tidak lagi berisi `test.txt`, dan search
mengembalikan hasil kosong.

Cek juga lewat UI: klik tombol "Delete" pada baris dokumen — harus hilang dari tabel setelah
konfirmasi.

## 8. Uji dokumen PDF (opsional, kalau kamu punya PDF)

```bash
curl -s -F "file=@contoh.pdf" http://localhost:8000/documents
curl -s "http://localhost:8000/search?q=<kata%20kunci%20dari%20isi%20pdf>"
```

**Ekspektasi:** teks berhasil diekstrak dan bisa dicari. Kalau PDF hasil scan (gambar, bukan teks
asli), `pypdf` tidak bisa OCR — hasil ekstraksi akan kosong dan upload akan gagal dengan error
"No extractable text in document". Ini limitasi yang perlu diketahui, bukan bug.

## 9. Uji koneksi MCP server dari client

Endpoint MCP: `http://localhost:8001/sse` (lokal) atau `https://<hostname-mcp-kamu>/sse` (VPS via
Cloudflare Tunnel).

- Cek endpoint hidup (harus menggantung/streaming, bukan error langsung close):
  ```bash
  curl -N --max-time 3 http://localhost:8001/sse
  ```
  **Ekspektasi:** koneksi terbuka dan diam (event stream), tidak ada error 404/500 instan.

- Uji sungguhan: hubungkan MCP client (Claude Desktop/Code, atau client lain yang kamu pakai)
  ke URL tersebut, lalu coba panggil tool:
  - `rag_list_documents()` → harus menampilkan dokumen yang sudah di-index.
  - `rag_search("kata kunci")` → harus mengembalikan chunk relevan.
  - `rag_add_note("judul", "isi catatan")` → cek lewat `GET /documents` atau UI, dokumen baru
    dengan nama `note:judul` harus muncul.

## 10. Uji restart & persistensi data

```bash
docker compose restart
sleep 10
curl -s http://localhost:8000/documents
```

**Ekspektasi:** dokumen yang sudah di-index sebelumnya tetap ada setelah restart (data tersimpan
di volume Docker `qdrant-data` dan `ardrag-data`, bukan hilang).

## 11. (Khusus VPS) Uji akses via Cloudflare Tunnel

Setelah `cloudflared` dikonfigurasi sesuai README:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://ardrag-ui.example.com/
curl -N --max-time 3 https://ardrag-mcp.example.com/sse
```

**Ekspektasi:** sama seperti uji lokal (status 200 untuk UI, stream terbuka untuk MCP), kali ini
lewat domain publik. Pastikan juga akses dari jaringan lain (bukan hanya dari VPS itu sendiri)
untuk memastikan tunnel benar-benar reachable dari luar.

## Ringkasan kriteria lulus

| # | Uji | Status yang diharapkan |
|---|---|---|
| 1 | Container up, tanpa error log | ✅ |
| 3 | UI & API dasar merespons | ✅ |
| 4-5 | Upload + search berhasil | ✅ |
| 6 | Replace mengganti isi lama sepenuhnya | ✅ |
| 7 | Delete membersihkan dari DB & vector store | ✅ |
| 9 | MCP tools bisa dipanggil dari client eksternal | ✅ |
| 10 | Data bertahan setelah restart | ✅ |
| 11 | Bisa diakses via domain publik Cloudflare | ✅ |
