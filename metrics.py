# calculate and save the values during process
def cal_metrics(level, rnd, hit, stat_dict, record_list, puz_id):
    if level not in stat_dict.keys():
        stat_dict[level] = {"round": 0.0, "hit": 0.0, "oa": 0.0, "num_sample": 0.0}

    # calculate
    stat_dict[level]["round"] += rnd
    stat_dict[level]["hit"] += hit
    stat_dict[level]["num_sample"] += 1
    # calculate overall (O/A) metric
    stat_dict[level]["oa"] += float(hit)/float(rnd)*100

    # record
    record_list.append([puz_id, level, hit, rnd])

    # current round and accuracy
    for key in stat_dict:
        avg_rnd = stat_dict[key]["round"]/stat_dict[key]["num_sample"]
        avg_acc = stat_dict[key]["hit"]/stat_dict[key]["num_sample"]
        avg_oa  = stat_dict[key]["oa"]/stat_dict[key]["num_sample"]
        print("level: %s, current num_sample: %s, avg round: %s, avg acc: %s, avg oa: %s"%(key, stat_dict[key]["num_sample"], avg_rnd, avg_acc, avg_oa))

    return stat_dict, record_list


# calculate the final metrics based on different difficulties
def cal_metrics_diff(stat_dict, record_list):
    # [num, acc_value, rnd_value, oa_value]
    overall_easy = [0, 0, 0, 0]
    overall_medium = [0, 0, 0, 0]
    overall_hard = [0, 0, 0, 0]

    # based on difficulty levels
    for item in record_list:
        cur_level = item[1]
        hit = item[2]
        rnd = item[3]
        if cur_level=='1' or cur_level=='2' or cur_level=='3':
            overall_easy[0] += 1
            overall_easy[1] += hit
            overall_easy[2] += rnd
            overall_easy[3] += float(hit)/float(rnd)*100
        elif cur_level=='4' or cur_level=='5' or cur_level=='6':
            overall_medium[0] += 1
            overall_medium[1] += hit
            overall_medium[2] += rnd
            overall_medium[3] += float(hit)/float(rnd)*100
        elif cur_level=='7' or cur_level=='8' or cur_level=='9':
            overall_hard[0] += 1
            overall_hard[1] += hit
            overall_hard[2] += rnd
            overall_hard[3] += float(hit)/float(rnd)*100
    
    # avoid there are no samples in each difficulty level
    if overall_easy[0]==0:
        overall_easy[0]+=1e-5
    if overall_medium[0]==0:
        overall_medium[0]+=1e-5
    if overall_hard[0]==0:
        overall_hard[0]+=1e-5

    # calculate the final evaluation metrics for each difficulty level
    esay_acc = overall_easy[1]/overall_easy[0]
    medium_acc = overall_medium[1]/overall_medium[0]
    hard_acc = overall_hard[1]/overall_hard[0]
    avg_acc = (esay_acc+medium_acc+hard_acc)/3.0
    print("[Final Acc] Easy: %.2f | Medium: %.2f | Hard: %.2f | Avg: %.2f"%(esay_acc, medium_acc, hard_acc, avg_acc))

    esay_rnd = overall_easy[2]/overall_easy[0]
    medium_rnd = overall_medium[2]/overall_medium[0]
    hard_rnd = overall_hard[2]/overall_hard[0]
    cnt = 0
    total_rnd = 0
    if esay_rnd!=0:
        cnt+=1
        total_rnd+=esay_rnd
    if medium_rnd!=0:
        cnt+=1
        total_rnd+=medium_rnd
    if hard_rnd!=0:
        cnt+=1
        total_rnd+=hard_rnd
    avg_rnd = float(total_rnd)/cnt
    print("[Final Rnd] Easy: %.2f | Medium: %.2f | Hard: %.2f | Avg: %.2f"%(esay_rnd, medium_rnd, hard_rnd, avg_rnd))

    esay_oa = overall_easy[3]/overall_easy[0]
    medium_oa = overall_medium[3]/overall_medium[0]
    hard_oa = overall_hard[3]/overall_hard[0]
    avg_oa = (esay_oa+medium_oa+hard_oa)/3.0
    print("[Final O/A] Easy: %.2f | Medium: %.2f | Hard: %.2f | Avg: %.2f"%(esay_oa, medium_oa, hard_oa, avg_oa))