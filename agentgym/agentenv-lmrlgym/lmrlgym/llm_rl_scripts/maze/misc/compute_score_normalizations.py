def compute_score(score, min_score, dataset_avg, max_score):
    if score < dataset_avg:
        return (score - min_score) / (dataset_avg - min_score) * 50
    else:
        return 50 + (score - dataset_avg) / (max_score - dataset_avg) * 50

maze_min_score = -101
maze_max_score = -6.84
maze_avg_score = -83

compute_score(-72.1, maze_min_score, maze_avg_score, maze_max_score)

# raw_scores = [-72.1, -56.4, -48.1, -6.97, -37.7, -71.755, -40.12]
# # raw_scores_po = [-79.5, -82.9, -80.3, -52.9, -91.7]
# for score in raw_scores:
#     print(compute_score(score, maze_min_score, maze_avg_score, maze_max_score))

chess_min_score = -401
chess_max_score = 1
chess_avg_score = 0.21

# chess_scores = [-22.3, -56.5, -28.2, -21.4, -16.0, -0.3]
# for score in chess_scores:
#     print("hello!")
#     print(compute_score(score, chess_min_score, chess_avg_score, chess_max_score))

endgames_min_score = -1
endgames_max_score = 1
endgames_avg_score = 0.586

endgames_scores = [0.112, -0.439, 0.588, 0.452, 0.814, -22.8, 0.149, 0.06976744186046512]
for score in endgames_scores:
    print(compute_score(score, endgames_min_score, endgames_avg_score, endgames_max_score))
    
text_nav_scores = []

# po_min_score = -101
# po_avg_score = -83
# po_max_score = -25.75

# po_scores = [-79.5, -82.9, -80.3, -52.9, -91.7, -71.04]
# for score in po_scores:
#     print(compute_score(score, po_min_score, po_avg_score, po_max_score))