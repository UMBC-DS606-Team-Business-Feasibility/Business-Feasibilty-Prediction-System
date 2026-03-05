def compare_cities(df):

    while True:

        city1 = input("\nCity 1: ").upper()
        city2 = input("City 2: ").upper()

        r1 = df[df["city"] == city1]
        r2 = df[df["city"] == city2]

        if r1.empty or r2.empty:
            print("City not found.")
            continue

        r1 = r1.iloc[0]
        r2 = r2.iloc[0]

        print("\nComparison")
        print("----------------")

        print(city1, "Score:", round(r1["feasibility_score"], 3))
        print(city2, "Score:", round(r2["feasibility_score"], 3))

        if r1["feasibility_score"] > r2["feasibility_score"]:
            print(city1, "is a better business location.")
        else:
            print(city2, "is a better business location.")

        break
