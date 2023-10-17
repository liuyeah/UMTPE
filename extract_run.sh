#!/usr/bin/env bash
python word_embedding/skip_gram_embedding.py

python patent/cpc/cpc_embedding/cpc_phrase_embedding.py
python patent/cpc/cpc_clustering/clustering.py

python patent/title/title_candidate/candidate_synthesis.py
python patent/title/title_embedding/title_phrase_embedding.py
python patent/title/title_graph/construct_graph.py
python patent/title/title_score/title_metrics.py
python patent/title/title_score/title_score_normalize.py
python patent/title/title_rank/title_rank.py
python patent/title/title_rank/title_to_text.py
python patent/title/title_clustering/title_clustering.py

python patent/abstract/abstract_candidate/candidate_synthesis.py
python patent/abstract/abstract_embedding/abstract_phrase_embedding.py
python patent/abstract/abstract_graph/construct_graph.py
python patent/abstract/abstract_score/abstract_metrics.py
python patent/abstract/abstract_score/abstract_score_normalize.py
python patent/abstract/abstract_rank/abstract_rank.py
python patent/abstract/abstract_rank/abstract_to_text.py
python patent/abstract/abstract_clustering/abstract_clustering.py

python patent/claim/claim_candidate/candidate_synthesis.py
python patent/claim/claim_embedding/claim_phrase_embedding.py
python patent/claim/claim_graph/construct_graph.py
python patent/claim/claim_score/claim_metrics.py
python patent/claim/claim_score/claim_score_normalize.py
python patent/claim/claim_rank/claim_rank.py
python patent/claim/claim_rank/claim_to_text.py

python result/select_phrase.py