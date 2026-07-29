from TinyStatistician import TinyStatistician


a = [1, 42, 300, 10, 59]
print(TinyStatistician().mean(a))
print(TinyStatistician().median(a))
print(TinyStatistician().quartile(a))
print(TinyStatistician().percentile(a, 10))
print(TinyStatistician().var(a))
print(TinyStatistician().std(a))

