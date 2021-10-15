from utils import tools

def count(dir, df):
    count = {}
    label_count = df.groupBy("label").count().orderBy('label').collect()
    count["clicked"] = label_count[0]["count"]
    count["non_clicked"] = label_count[1]["count"]
    count["total"] = df.count()
    tools.makedir(dir)
    tools.write_json(dir + "count.json", count)

def stats(dir, df, columns):
    def get_mode(df, c):
        frequency = df.groupBy(c).count().orderBy("count", ascending = False)
        null_count = frequency.filter(df[c].isNull()).collect()
        if null_count:
            null_cnt = null_count[0]["count"]
        else:
            null_cnt = 0
        for row in frequency.head(2):
            if row[c] == None:
                continue
            mode = row[c]
            break
        return int(mode), null_cnt
    
    def key(stat_type, c):
        return "{}({})".format(stat_type, c)

    stats = {}
    _, mean, stddev, mn, _, median, _, mx = df.summary().collect()
    for c in columns:
        stats[key("min", c)] = int(mn[c])
        stats[key("max", c)] = int(mx[c])
        stats[key("avg", c)] = round(float(mean[c]), 2)
        stats[key("median", c)] = int(median[c])
        stats[key("mode", c)], stats[key("nulls", c)] = get_mode(df, c)
        stats[key("stddev", c)] = round(float(stddev[c]), 2)
    tools.makedir(dir)
    tools.write_json(dir + "stats.json", stats)


def vocabs(dir, df, columns, word_size = 8):
    tools.makedir(dir)
    for c in columns:
        vocab_list = df.select(c).distinct().toPandas()
        while True:
            null_feature = tools.generate_cat(word_size)
            if null_feature not in vocab_list:
                vocab_list = vocab_list.append({c : null_feature}, ignore_index=True)
                break
        vocab_dir = dir + c + "/"
        tools.makedir(vocab_dir)
        vocab_list.to_csv(vocab_dir + "vocabs.csv", header = False, index = False)
        count = {"count" : len(vocab_list), "null_feature" : null_feature}
        tools.write_json(vocab_dir + "count.json", count)