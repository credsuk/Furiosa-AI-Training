scores = [85, 45, 70, 55, 95]

results = list(map(lambda x: "합격" if x >= 60 else "불합격", scores))

print(results)  # ['합격', '불합격', '합격', '불합격', '합격']