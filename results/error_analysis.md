NDCG@10 on test queries by query type (rule-based tags, a query can have several).

| query type | n | bm25_fields | dense_vecdex | hybrid_rrf | rerank_ms-marco-MiniLM-L-6-v2 | lambdamart |
|---|---|---|---|---|---|---|
| len_1_2 | 45 | 0.749 | 0.761 | 0.776 | 0.780 | 0.792 |
| len_3_4 | 83 | 0.780 | 0.738 | 0.797 | 0.802 | 0.814 |
| len_5_plus | 16 | 0.769 | 0.719 | 0.778 | 0.787 | 0.798 |
| colour | 12 | 0.707 | 0.590 | 0.739 | 0.756 | 0.774 |
| material | 9 | 0.736 | 0.564 | 0.709 | 0.824 | 0.767 |
| style | 11 | 0.835 | 0.783 | 0.856 | 0.863 | 0.888 |
| size_or_number | 20 | 0.696 | 0.694 | 0.723 | 0.706 | 0.702 |
| any_attribute | 50 | 0.734 | 0.673 | 0.753 | 0.767 | 0.768 |
| rare_word | 24 | 0.754 | 0.665 | 0.741 | 0.762 | 0.769 |
| plain_type | 74 | 0.801 | 0.819 | 0.831 | 0.825 | 0.846 |

Worst test queries for lambdamart (E = Exact, P = Partial, I = Irrelevant, ? = unjudged):

- **dull bed with shirt head board** (class ''; 0 Exact, 34 Partial judged) NDCG@10 = 0.000
    - [?] adjustable head bed with remote (Adjustable Beds)
    - [?] full over full bunk bed (Kids Beds)
    - [?] house of doolittle all-purpose/vacation plan-a-board wall mounted calendar board (Memo Boards)
    - [?] corkboard/bulletin board (Memo Boards)
    - [?] wall mounted bulletin board (Memo Boards)
- **living room ideas** (class 'Living Room Sets'; 16 Exact, 5 Partial judged) NDCG@10 = 0.000
    - [?] harrad 3 piece living room set (Living Room Sets)
    - [?] darl 2 piece living room set (Living Room Sets)
    - [?] dehaan mid-century 3 piece living room set (Living Room Sets)
    - [?] living room floor (Game Chairs)
    - [?] addyson living room 3 piece coffee table set (Living Room Table Sets)
- **rack glass** (class 'Candle Holders'; 2 Exact, 432 Partial judged) NDCG@10 = 0.000
    - [?] bernon solid wood wall mounted wine glass rack (Wine Racks)
    - [?] bernardo solid wood wall mounted wine glass rack (Wine Racks)
    - [?] hanging wine glass rack (Wine Racks)
    - [?] kena hanging wine glass rack (Wine Racks)
    - [?] podgorni hanging wine glass rack (Wine Racks)
- **large bases** (class 'Patio Umbrella Stands & Bases'; 0 Exact, 1 Partial judged) NDCG@10 = 0.000
    - [?] large table base (Lighting Accessories)
    - [?] preside base (Furniture Parts)
    - [?] 30 '' cabinet vanity base (Vanity Bases)
    - [?] 15 '' cabinet vanity base (Vanity Bases)
    - [?] 18 '' bathroom vanity base only (Vanity Bases)
- **outdoor waterproof chest** (class 'Deck Boxes'; 46 Exact, 40 Partial judged) NDCG@10 = 0.136
    - [I] outdoor waterproof chaise lounge cover (Furniture Covers)
    - [P] waterproof 1 hour fire chest 0.46 cuft (Safes)
    - [I] english garden outdoor waterproof patio chair cover with 2 year warranty (Furniture Covers)
    - [?] waterproof garden round breathable patio table cover (Furniture Covers)
    - [I] outdoor waterproof camping 4 person tent (Camping Tents & Shelters)
- **pedistole sink** (class 'Kitchen Sinks'; 0 Exact, 152 Partial judged) NDCG@10 = 0.145
    - [?] matte stone oval vessel bathroom sink (Bathroom Sinks)
    - [?] rectangular undermount lavatory sink (Bathroom Sinks)
    - [?] ceramic rectangular vessel bathroom sink (Bathroom Sinks)
    - [?] vigo matte stone™ rectangular vessel bathroom sink (Bathroom Sinks)
    - [?] white ceramic rectangular vessel bathroom sink (Bathroom Sinks)
- **oliver parsons** (class 'Dining Chairs'; 0 Exact, 86 Partial judged) NDCG@10 = 0.161
    - [?] albertas upholstered parsons chair in gray (Accent Chairs|Dining Chairs)
    - [?] oecusse upholstered parsons chair in gray (Accent Chairs|Dining Chairs)
    - [?] alexious upholstered parsons chair in gray (Accent Chairs|Dining Chairs)
    - [P] nadia upholstered parsons chair (Accent Chairs|Dining Chairs)
    - [?] chesica upholstered parsons chair (Accent Chairs|Dining Chairs)
- **foutains with brick look** (class 'Wallpaper'; 12 Exact, 78 Partial judged) NDCG@10 = 0.161
    - [?] brick 2 '' x 10 '' porcelain subway tile (Floor & Wall Tile)
    - [P] 8 '' x 2.5 '' brick look subway tile (Floor & Wall Tile)
    - [P] citadel 3 '' x 6 '' ceramic brick look subway tile (Floor & Wall Tile)
    - [I] tribeca 2 '' x 10 '' porcelain brick look wall & floor tile (Floor & Wall Tile)
    - [?] brick 3 '' x 8 '' cement subway tile (Floor & Wall Tile)
- **desk for kids** (class 'Kids Desks'; 78 Exact, 919 Partial judged) NDCG@10 = 0.222
    - [?] kids writing table with reading light , height adjustable functional student study desk and chair set for children boys girls ()
    - [?] kids desk and chair set , height adjustable children study table students interactive workstation , with wood tiltable anti-reflective tabletop and pull-out drawer for school boys & girls (Kids Desks)
    - [?] kids desk and chair set , height adjustable school children study table with led lamp , bookstand , wood tilting tabletop , drawer storage , for 3-15 years old students (Kids Desks)
    - [?] kids desk and chair set , height adjustable children study table with wood tiltable anti-reflective tabletop , bookstand , pull-out drawer storage and touch led for school students (Kids Desks)
    - [E] kids wrting desk wood ()
- **toddler couch fold out** (class 'Kids Chairs'; 3 Exact, 72 Partial judged) NDCG@10 = 0.275
    - [?] convertible toddler bed (Kids Beds)
    - [P] basel 3 in 1 fold out monkey kids recliner (Kids Chairs)
    - [P] bernartice oxygen fold out kids cotton sleeper (Kids Chairs)
    - [?] perch toddler bed (Kids Beds)
    - [P] kids single armrest couch sofa with ottoman (Kids Chairs)
- **navy and peach bedding** (class 'Bedding Sets'; 0 Exact, 115 Partial judged) NDCG@10 = 0.298
    - [P] navy blue and pink watercolor floral 5 piece crib bedding set (Crib Bedding Sets)
    - [?] geometric 7 piece bed-in-a-bag reversible comforter (Bedding Sets)
    - [?] waverly place little dinosaurs 8 piece crib bedding set (Crib Bedding Sets)
    - [?] mod dinosaur 4 piece crib bedding set (Crib Bedding Sets)
    - [?] connollee 6 piece crib bedding set (Crib Bedding Sets)
- **70s inspired furniture** (class 'Wall Art'; 20 Exact, 87 Partial judged) NDCG@10 = 0.362
    - [P] old fashioned 60s 70s inspired geometric sheet set (Sheets And Sheet Sets)
    - [I] 60s-70s pop art-inspired green area rug (Area Rugs)
    - [P] old fashioned 60s 70s inspired polka dot fitted sheet (Sheets And Sheet Sets)
    - [P] 60s 70s pop art inspired circles vintage design art duvet cover set (Bedding Sets)
    - [P] 70s party party flyer inspired and curved with daisies audio cassette tape duvet cover set (Bedding Sets)
