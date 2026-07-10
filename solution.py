import json
import subprocess

with open("q-youtube-metadata-filter-server.json", "r") as f:
    cfg = json.load(f)

videos = []

for url in cfg["source_urls"]:
    try:
        out = subprocess.check_output(
            ["yt-dlp", "--dump-json", url],
            text=True
        )
        meta = json.loads(out)

        duration = meta.get("duration", 0)
        if not (cfg["min_duration_seconds"] <= duration <= cfg["max_duration_seconds"]):
            continue

        title = meta.get("title", "")
        description = meta.get("description", "")

        combined = (title + " " + description).lower()

        if not all(word.lower() in combined for word in cfg["required_words"]):
            continue

        if any(word.lower() in combined for word in cfg["forbidden_words"]):
            continue

        videos.append({
            "url": url,
            "upload_date": meta.get("upload_date", ""),
            "id": meta.get("id", "")
        })

    except Exception:
        pass

videos.sort(key=lambda x: (-int(x["upload_date"]), x["id"]))

result = {
    "urls": [v["url"] for v in videos[:cfg["limit"]]]
}

with open("output.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))