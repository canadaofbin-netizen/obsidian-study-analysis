---
tags: [dashboard]
---
# 📊 Master Study Analysis Dashboard

Welcome to the automated Zettelkasten dashboard. This page uses Dataview Queries to automatically track your knowledge base under the **2-Track System**.

## 📝 1. Fact Notes (Track 1)
Objective data extracted from papers, tagged with `#fact_note`.
```dataview
TABLE tags AS "Tags", file.ctime AS "Extracted On"
FROM "wiki"
WHERE contains(tags, "fact_note")
SORT file.ctime DESC
```

## 🧠 2. Correlation Concept Hubs (Track 2)
Intersections and latent correlations discovered by the 100-round agent debates.
```dataview
TABLE tags AS "Tags", file.mtime AS "Last Modified"
FROM "wiki/concepts"
```

## 👤 3. Entities (Authors, Organizations, Tools)
Concrete entities mapped in the vault.
```dataview
TABLE tags AS "Tags"
FROM "wiki/entities"
```

## ⚔️ 4. Synthesis & Debates
Multi-agent debate transcripts and executive correlation reports.
```dataview
TABLE file.ctime AS "Created"
FROM "wiki/synthesis"
```
