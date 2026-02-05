import argparse
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def unbiased_pass_at_k_accuracy(file_path, k, n):
    # Read the JSONLines file
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Group predictions by question_id
    grouped_data = defaultdict(list)
    for example in data:
        question_id = example["question_id"]
        if len(grouped_data[question_id]) >= n:
            continue
        grouped_data[question_id].append(example)

    # Calculate unbiased pass@K accuracy
    total_pass_k_prob = 0
    total_questions = 0
    
    for question_id, examples in grouped_data.items():
        # Count correct and total solutions for this question
        assert len(examples) == n  # Total number of samples
        c = sum(1 for ex in examples if ex["label"])  # Number of correct samples
        # Apply unbiased pass@k formula
        assert n >= k
        # Calculate unbiased pass@k probability
        if n - c < k:
            prob = 1.0
        else:
            prob = 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
            
        total_pass_k_prob += prob
        total_questions += 1
    
    # Calculate average pass@K probability across all questions
    avg_pass_k = total_pass_k_prob / total_questions if total_questions > 0 else 0
    print(f"Questions Number: {total_questions}, Unbiased Pass@{k}/{n}: {avg_pass_k}")
    return avg_pass_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the JSONLines file")
    args = parser.parse_args()
   
    Ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    print(f"Calculating Unbiased Pass@K for {args.file_path}")
    for K in tqdm(Ks):
        print("-" * 80)
        test_file = args.file_path
        unbiased_pass_k_accuracy = unbiased_pass_at_k_accuracy(test_file, k=K, n=256)
    #AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #MATH-test-temp_0.6-top_p_0.95-top_k_-1.jsonl

    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp1-0_1/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.09869791666666661  Pass@2/256: 0.1496446078431372 Pass@4/256: 0.20100488441618597 Pass@8/256: 0.24805148512059544 Pass@16/256: 0.2974217305582118 Pass@32/256: 0.36149404545239616 Pass@64/256: 0.44887731538237574 Pass@128/256: 0.5548543713407353 Pass@256/256: 0.6666666666666666
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.10260416666666662  Pass@2/256: 0.14319342320261436 Pass@4/256: 0.1865691728591471 Pass@8/256: 0.23111731387730164 Pass@16/256: 0.2745886089133048 Pass@32/256: 0.3178090954917093 Pass@64/256: 0.3606816480605272 Pass@128/256: 0.40837367898370547 Pass@256/256: 0.4666666666666667
    ##python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp1-10/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.09648437499999994  Pass@2/256: 0.13632250816993458 Pass@4/256: 0.1709422038593845 Pass@8/256: 0.20303316110378622 Pass@16/256: 0.236684305867967 Pass@32/256: 0.27046849871689294 Pass@64/256: 0.30914593472442864 Pass@128/256: 0.3584960736201843 Pass@256/256: 0.4
    #python3 calculate_metrics.py --file_path=gen_256/Qwen2.5-Math-7B/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.06080729166666661 Pass@2/256: 0.09876736111111106 Pass@4/256: 0.14406295406183386 Pass@8/256: 0.19299141528729133 Pass@16/256: 0.244412812536658 Pass@32/256: 0.2913797644259507 Pass@64/256: 0.3343945992755238 Pass@128/256: 0.3917584848230792 Pass@256/256: 0.4666666666666667

    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.10260416666666662  Pass@2/256: 0.14319342320261436 Pass@4/256: 0.1865691728591471 Pass@8/256: 0.23111731387730164 Pass@16/256: 0.2745886089133048 Pass@32/256: 0.3178090954917093 Pass@64/256: 0.3606816480605272 Pass@128/256: 0.40837367898370547 Pass@256/256: 0.4666666666666667
    #python3 calculate_metrics.py --file_path=gen_256/Qwen2.5-Math-7B/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.06080729166666661 Pass@2/256: 0.09876736111111106 Pass@4/256: 0.14406295406183386 Pass@8/256: 0.19299141528729133 Pass@16/256: 0.244412812536658 Pass@32/256: 0.2913797644259507 Pass@64/256: 0.3343945992755238 Pass@128/256: 0.3917584848230792 Pass@256/256: 0.4666666666666667
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp2-p/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.09817708333333328  Pass@2/256: 0.1450357434640522 Pass@4/256: 0.19448872103539372 Pass@8/256: 0.2431227090229414 Pass@16/256: 0.287891279771253 Pass@32/256: 0.32988073623741787 Pass@64/256: 0.3713461121976246 Pass@128/256: 0.41237344548052673 Pass@256/256: 0.4666666666666667
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp2-p2/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.1006510416666666  Pass@2/256: 0.14545853758169933 Pass@4/256: 0.19256534047047597 Pass@8/256: 0.2394656669518827 Pass@16/256: 0.2844693227180926 Pass@32/256: 0.3296134491087603 Pass@64/256: 0.3763169924477764 Pass@128/256: 0.42253993261187456 Pass@256/256: 0.4666666666666667
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp2-1_p/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.09713541666666661  Pass@2/256: 0.1426521650326797 Pass@4/256: 0.19208988128256813 Pass@8/256: 0.24412298024314605 Pass@16/256: 0.2968360629029561 Pass@32/256: 0.3516189156657002 Pass@64/256: 0.4114264043813084 Pass@128/256: 0.48733469617292374 Pass@256/256: 0.6
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp3-0_1_1_p/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.08958333333333325  Pass@2/256: 0.13657373366013068 Pass@4/256: 0.18406544253426996 Pass@8/256: 0.22700396581933452 Pass@16/256: 0.2664277181213125 Pass@32/256: 0.30941311271597444 Pass@64/256: 0.3691923748728121 Pass@128/256: 0.44814513393532385 Pass@256/256: 0.5333333333333333

    ##python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp1-10/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.09648437499999994  Pass@2/256: 0.13632250816993458 Pass@4/256: 0.1709422038593845 Pass@8/256: 0.20303316110378622 Pass@16/256: 0.236684305867967 Pass@32/256: 0.27046849871689294 Pass@64/256: 0.30914593472442864 Pass@128/256: 0.3584960736201843 Pass@256/256: 0.4
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp5_mixed/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.0986979166666666  Pass@2/256: 0.14124489379084965 Pass@4/256: 0.18641345501351392 Pass@8/256: 0.23165211297607405 Pass@16/256: 0.27486546402934303 Pass@32/256: 0.32463167414247224 Pass@64/256: 0.38982064347951495 Pass@128/256: 0.46972340010514263 Pass@256/256: 0.5666666666666667
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp4_sqrt1-p2/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.09648437499999994  Pass@2/256: 0.13903186274509796 Pass@4/256: 0.1862378080297508 Pass@8/256: 0.23714029447221507 Pass@16/256: 0.2861690231597592 Pass@32/256: 0.3310962830056288 Pass@64/256: 0.3756891922897333 Pass@128/256: 0.43135126789183303 Pass@256/256: 0.5
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp4_sqrtp/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.10390624999999996  Pass@2/256: 0.14585477941176464 Pass@4/256: 0.19126093924778526 Pass@8/256: 0.23860915750057796 Pass@16/256: 0.2883842171708407 Pass@32/256: 0.3433634430472791 Pass@64/256: 0.3971597599878186 Pass@128/256: 0.4459577223717119 Pass@256/256: 0.5

    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp5_mixed_reverse/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.10507812499999993  Pass@2/256: 0.14809538398692804 Pass@4/256: 0.1941400496039192 Pass@8/256: 0.24178154404482033 Pass@16/256: 0.28913263091749386 Pass@32/256: 0.340359661478461 Pass@64/256: 0.40235380895388656 Pass@128/256: 0.47803764045151376 Pass@256/256: 0.5666666666666667
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp5_mixed_reverse/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.60986328125  Pass@2/256: 0.6810140931372547 Pass@4/256: 0.7335862950236348 Pass@8/256: 0.7739367200845289 Pass@16/256: 0.8131234388184986 Pass@32/256: 0.8531385940048308 Pass@64/256: 0.8916908680405436 Pass@128/256: 0.924319773154435 Pass@256/256: 0.95
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp5_mixed2/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.09986979166666662  Pass@2/256: 0.14387969771241824 Pass@4/256: 0.1926130789412338 Pass@8/256: 0.245134570216948 Pass@16/256: 0.2987570376016484 Pass@32/256: 0.3538669437049774 Pass@64/256: 0.4086618193180917 Pass@128/256: 0.4656587158326148 Pass@256/256: 0.5333333333333333
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp5_mixed2/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.61015625  Pass@2/256: 0.6924808517156862 Pass@4/256: 0.7598774872614774 Pass@8/256: 0.8122819810280448 Pass@16/256: 0.8534342323390763 Pass@32/256: 0.8881038739495437 Pass@64/256: 0.9203100298339933 Pass@128/256: 0.9514806672790044 Pass@256/256: 0.975
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp5_mixed_withneg2/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.6126953125  Pass@2/256: 0.6939192708333334 Pass@4/256: 0.764430362885989 Pass@8/256: 0.8304729214761994 Pass@16/256: 0.884885113643396 Pass@32/256: 0.9225400330565978 Pass@64/256: 0.9459651861741522 Pass@128/256: 0.9609440711652933 Pass@256/256: 0.975
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp5_mixed_withneg2/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.10598958333333328  Pass@2/256: 0.15223141339869276 Pass@4/256: 0.19996518407945169 Pass@8/256: 0.2492057132767008 Pass@16/256: 0.29901955821632176 Pass@32/256: 0.3483546311189485 Pass@64/256: 0.39923448668310385 Pass@128/256: 0.45620160426864287 Pass@256/256: 0.5333333333333333

    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp1-0_1/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.609375  Pass@2/256: 0.6973314950980392 Pass@4/256: 0.7664119436035753 Pass@8/256: 0.8218409438242016 Pass@16/256: 0.8653981927866601 Pass@32/256: 0.9060314975273279 Pass@64/256: 0.9438524931477884 Pass@128/256: 0.9727187909599081 Pass@256/256: 1.0
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.5921875  Pass@2/256: 0.666734068627451 Pass@4/256: 0.7209644711013005 Pass@8/256: 0.7641959447095317 Pass@16/256: 0.8063063222183693 Pass@32/256: 0.8482249590178326 Pass@64/256: 0.8829807826265872 Pass@128/256: 0.9076953529363083 Pass@256/256: 0.925
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp1-10/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.6001953125  Pass@2/256: 0.6572878370098039 Pass@4/256: 0.7028893268904228 Pass@8/256: 0.7387881654246918 Pass@16/256: 0.7754426516882653 Pass@32/256: 0.8165143005781615 Pass@64/256: 0.8618138641392143 Pass@128/256: 0.9085453511643753 Pass@256/256: 0.95
    #python3 calculate_metrics.py --file_path=gen_256/Qwen2.5-Math-7B/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.3998046875 Pass@2/256: 0.5529097732843137 Pass@4/256: 0.6863022765146174 Pass@8/256: 0.7858797156072729 Pass@16/256: 0.8495610938266012 Pass@32/256: 0.8940723927445771 Pass@64/256: 0.9335675194691297 Pass@128/256: 0.9735287342342236 Pass@256/256: 1.0

    #python3 calculate_metrics.py --file_path=gen_256/Qwen2.5-Math-7B/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.3998046875 Pass@2/256: 0.5529097732843137 Pass@4/256: 0.6863022765146174 Pass@8/256: 0.7858797156072729 Pass@16/256: 0.8495610938266012 Pass@32/256: 0.8940723927445771 Pass@64/256: 0.9335675194691297 Pass@128/256: 0.9735287342342236 Pass@256/256: 1.0
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.5921875  Pass@2/256: 0.666734068627451 Pass@4/256: 0.7209644711013005 Pass@8/256: 0.7641959447095317 Pass@16/256: 0.8063063222183693 Pass@32/256: 0.8482249590178326 Pass@64/256: 0.8829807826265872 Pass@128/256: 0.9076953529363083 Pass@256/256: 0.925
    # python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp2-1_p/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.6078125 Pass@2/256: 0.6989537377450981  Pass@4/256: 0.7761138757043774 Pass@8/256: 0.8363175783369365 Pass@16/256: 0.8785038349334837 Pass@32/256: 0.9088110175253545 Pass@64/256: 0.9317143766705678 Pass@128/256: 0.946364192721383 Pass@256/256: 0.95
    # python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp2-p/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.6153320312499999 Pass@2/256: 0.7080897671568628  Pass@4/256: 0.7857559205582112 Pass@8/256: 0.8446006976497094 Pass@16/256: 0.887779512889196 Pass@32/256: 0.9225976802725556 Pass@64/256: 0.9550719594982467 Pass@128/256: 0.9823382614882383 Pass@256/256: 1.0
    # python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp2-p2/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.6189453125 Pass@2/256: 0.6941122855392157  Pass@4/256: 0.7532851017067995 Pass@8/256: 0.8030489468633917 Pass@16/256: 0.8536486741883189 Pass@32/256: 0.9036696225818279 Pass@64/256: 0.940939252291473 Pass@128/256: 0.9613022505426938 Pass@256/256: 0.975
    # python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp3-0_1_1_p/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.5953124999999999 Pass@2/256: 0.6929250919117645  Pass@4/256: 0.7717139061175573 Pass@8/256: 0.8360028602823387 Pass@16/256: 0.8833857977742611 Pass@32/256: 0.921215426127058 Pass@64/256: 0.9557166820710703 Pass@128/256: 0.9866676746560523 Pass@256/256: 1.0

    # python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp5_mixed/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.60322265625 Pass@2/256: 0.6959329044117647  Pass@4/256: 0.7753202411726261 Pass@8/256: 0.8372010522638854 Pass@16/256: 0.8828022277449377 Pass@32/256: 0.9199779748485986 Pass@64/256: 0.9508300783453769 Pass@128/256: 0.9702017032410248 Pass@256/256: 0.975
    # python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp4_sqrt1-p2/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.5963867187499999 Pass@2/256: 0.6787584252450981  Pass@4/256: 0.7397693479828442 Pass@8/256: 0.7897145122589331 Pass@16/256: 0.8374095593914047 Pass@32/256: 0.8838986337254771 Pass@64/256: 0.9244019791206393 Pass@128/256: 0.9578268665383405 Pass@256/256: 0.95
    # python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp4_sqrtp/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.61875 Pass@2/256: 0.6937055759803921  Pass@4/256: 0.753341554026531 Pass@8/256: 0.8036590337266315 Pass@16/256: 0.8484929704970277 Pass@32/256: 0.8906608914151037 Pass@64/256: 0.9273242321880992 Pass@128/256: 0.9592017848949321 Pass@256/256: 1.0


    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp9-1_slow/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.608 Pass@2/256: 0.697  Pass@4/256: 0.764 Pass@8/256: 0.818 Pass@16/256: 0.860 Pass@32/256: 0.891 Pass@64/256: 0.918 Pass@128/256: 0.946 Pass@256/256: 0.975
    #python3 calculate_metrics.py --file_path=gen_256/MATH-Qwen2.5-Math-7B-GRPO-exp9-1_slow/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #Pass@1/256: 0.108 Pass@2/256: 0.153  Pass@4/256: 0.201 Pass@8/256: 0.251 Pass@16/256: 0.302 Pass@32/256: 0.352 Pass@64/256: 0.400 Pass@128/256: 0.445 Pass@256/256: 0.5


    #python3 calculate_metrics.py --file_path=gen_256/DeepSeek-R1-7B/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    # 64.2 &73.0 &80.1 &85.5 &89.8 &92.8 &95.2 &97.4 &100
    #python3 calculate_metrics.py --file_path=gen_256/DeepSeek-R1-7B/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    # 22.7 &26.5 &29.7 &33.2 &37.8 &43.2 &48.0 &51.1 &53.3
    #python3 calculate_metrics.py --file_path=gen_256/DeepSeek-R1-7B/MATH-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #84.3&89.6&92.8&94.8&96.0&96.8&97.4&97.8&98.0

    #python3 calculate_metrics.py --file_path=gen_256/DeepSeek-R1-7B-grpo/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    # 82.8 & 89.3 & 92.3 & 94.0 & 95.2 & 96.1 & 97.2 & 98.8 & 100
    #python3 calculate_metrics.py --file_path=gen_256/DeepSeek-R1-7B-grpo/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    # 28.5 & 35.0 & 41.4 & 47.7 & 53.1 & 57.6 & 62.1 & 67.2 & 73.3
    #python3 calculate_metrics.py --file_path=gen_256/DeepSeek-R1-7B-grpo/MATH-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    # 92.4 & 95.3 & 96.5 & 97.0 & 97.3 & 97.7 & 98.0 & 98.3 & 98.6

    #python3 calculate_metrics.py --file_path=gen_256/DeepSeek-R1-7B-agrae/AMC23-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    #83.2 & 89.6 & 92.5 & 94.4 & 95.6 & 96.7 & 98.0 & 99.3 & 100
    #python3 calculate_metrics.py --file_path=gen_256/DeepSeek-R1-7B-agrae/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    # 29.4 & 36.0 & 42.6 & 48.9 & 54.0 & 58.3 & 63.0 & 68.3 & 74.3
    #python3 calculate_metrics.py --file_path=gen_256/DeepSeek-R1-7B-agrae/MATH-test-temp_0.6-top_p_0.95-top_k_-1.jsonl
    # 91.9 & 94.7 & 95.9 & 96.5 & 96.9 & 97.3 & 97.6 & 98.0 & 98.4



