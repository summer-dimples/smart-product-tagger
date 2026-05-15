# smart-product-tagger
Zero-human-label LLM tagging + similarity retrieval pipeline for e-commerce catalogs

> An end-to-end pipeline that turns raw product descriptions into standardized tags, dense embeddings, and similarity recommendations — without any human labelling. Designed to bootstrap recommendation, search, and ads cold-start for long-tail SKUs.


---

##  Why this exists

E-commerce platforms have **millions of long-tail SKUs without structured tags**. This hurts:
- search recall (no tags = no inverted-index hits),
- recommendation cold-start (new items have no behavioral signal),
- ad targeting (no attributes = no audience matching).

Manual tagging is expensive. This repo implements an alternative: **let an LLM tag freely, then standardise the tag space, embed, and use an LLM Critic to generate the ground truth needed to fine-tune the retrieval encoder.**

---

##  Pipeline architecture

```
[Raw product descriptions]
          ↓
  ┌─────────────────────────┐
  │ 1. LLM Free Tagging     │   ← OpenRouter, batch-with-ID-mapping
  └─────────────────────────┘
          ↓ tagged_all.csv
  ┌─────────────────────────┐
  │ 2. Hierarchical         │   ← Sentence-Transformers + Ward linkage
  │    Clustering &         │     auto-tuned silhouette threshold
  │    Standardisation      │     optional LLM consistency check
  └─────────────────────────┘
          ↓ clusters_all.csv + tag_mapping.json
  ┌─────────────────────────┐
  │ 3. Autoencoder Encoder  │   ← Multi-hot input → 6-dim dense latent
  └─────────────────────────┘
          ↓ embeddings_ae.npy
  ┌─────────────────────────┐
  │ 4. Cosine Retrieval     │   ← For 100 random toys, return Top-10
  └─────────────────────────┘
          ↓ retrieval_results.json
  ┌─────────────────────────┐
  │ 5. LLM Critic Labelling │   ← qwen-flash, async concurrent
  │    (SIMILAR/DISSIMILAR) │     produces "ground truth" for free
  └─────────────────────────┘
          ↓ critic_labels.json
  ┌─────────────────────────┐
  │ 6. Siamese Fine-tuning  │   ← Contrastive loss, margin=1.0
  └─────────────────────────┘
          ↓ embeddings_siamese.npy
  ┌─────────────────────────┐
  │ 7. Evaluation           │   ← Precision@K, Recall@K (K=1,3,5,10)
  └─────────────────────────┘
```

---

##  Results

On the Toys subset (100 random test items, evaluated against LLM Critic labels):

| Embedding strategy            | P@10  | R@10  |
|-------------------------------|-------|-------|
| Raw free tags → autoencoder   | _baseline_ | _baseline_ |
| Standardised tags → autoencoder | ▲ improved | ▲ improved |
| Siamese fine-tuned (proposed) | ▲▲ best    | ▲▲ best    |

> *Numbers in `output/phase4_critic_eval/evaluation_report.md`. Standardisation alone gives a measurable lift; Siamese fine-tuning adds another margin.*

**Cost profile**: Tagging stays under **USD 0.5 per 1,000 SKUs** using OpenRouter's free / cheap tier; Critic labelling is also bounded — for an entire 100×10 retrieval evaluation, total LLM spend was under **USD 5**.

---

##  Quick start

```bash
git clone https://github.com/yixuanhuang/smart-product-tagger
cd smart-product-tagger
pip install -r requirements.txt

# put your OpenRouter key in .env
echo "OPENROUTER_API_KEY=sk-or-..." > .env

# put your product list at data/Dataset/product_list.csv
# minimum schema: id, description, brand, product_type

# run the full pipeline
python tagger.py            # phase 1
python cluster_tags.py      # phase 2
python build_embeddings.py  # phase 3
python build_embeddings_raw.py  # phase 3 ablation
python retrieval.py         # phase 4
python critic.py            # phase 4
python siamese.py           # phase 5
python evaluation.py        # phase 6
```

Outputs at each phase land under `output/phase{N}_*/` for inspection.

---

##  Configuration knobs

| File                       | Knob                                  | Effect                                                  |
|----------------------------|---------------------------------------|---------------------------------------------------------|
| `tagger.py`                | `BATCH_SIZE`, `max_tags_per_item`     | Throughput vs cost                                      |
| `cluster_tags.py`          | silhouette threshold range, `MAX_LLM_CALLS` | Tag granularity & LLM consistency check spend           |
| `autoencoder.py`           | `latent_dim`                          | Retrieval embedding dimensionality                      |
| `critic.py`                | concurrency limit (default 15)        | Critic labelling speed vs API rate limits               |
| `siamese.py`               | `margin`, epochs, lr                  | Contrastive learning aggressiveness                     |

---

##  Ablation: why standardise tags?

`build_embeddings_raw.py` is the deliberate baseline — same autoencoder, but trained on **raw free tags** instead of the cluster-standardised vocabulary. The numerical lift you see between this baseline and the standardised version is the value of phase 2 (hierarchical clustering). Skipping standardisation produces a tag vocabulary so noisy that the autoencoder cannot recover useful structure.

---

##  Productionisation directions

This repo is intentionally a research pipeline, not a deployed service. Sketches for going further:
- Wrap phases 1+2+3 as a FastAPI endpoint that ingests product feeds (CSV/JSON) and returns standardised tags + dense embeddings.
- Use phase 4-6 as an offline weekly job to refresh the retrieval encoder against drifting taxonomy.
- Add a Gradio / Streamlit front-end for catalog managers to spot-check tag quality.
- Plug into existing vector DBs (Pinecone, Milvus, pgvector) for the retrieval index.

---

##  Tech stack

- **LLM API**: OpenRouter (`inclusionai/ling-2.6-flash:free` for tagging, `qwen/qwen-flash` for Critic)
- **Text embeddings**: Sentence-Transformers `all-MiniLM-L6-v2`
- **Clustering**: `scipy.cluster.hierarchy` with Ward linkage; silhouette-tuned cutoff
- **Deep learning**: PyTorch autoencoder (multi-hot in → 6-dim dense → reconstruct), Siamese contrastive head
- **Evaluation**: cosine similarity, Precision@K, Recall@K


---

##  License

MIT. See LICENSE.

##  Author

**Yixuan Huang**  [email](mailto:yixuanhuang102171@outlook.com)
