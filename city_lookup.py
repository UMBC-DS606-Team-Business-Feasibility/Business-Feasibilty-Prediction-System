def lookup_city_score(df):

    while True:

        city = input("\nEnter a city name (or type exit): ").strip().upper()

        if city == "EXIT":
            break

        result = df[df["city"].str.upper() == city]

        if result.empty:
            print("City not found.")
        else:
            row = result.iloc[0]

            print("\nBusiness Feasibility Score")
            print("--------------------------")
            print("City:", row["city"])
            print("Score:", round(row["feasibility_score"], 3))
            print("Median Income:", row["median_income"])
            print("Employment Rate:", row["employment_rate"])
