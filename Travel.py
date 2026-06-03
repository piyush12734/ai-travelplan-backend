def get_city_info(city_name, days):
    city = {
        "name": city_name,
        "days": days,
        "activities": [
            f"Explore the old town of {city_name}",
            f"Try local street food in {city_name}",
            f"Visit the top museum in {city_name}",
            f"Walk through the markets of {city_name}",
            f"Take a sunset tour in {city_name}"
        ],
        "tip": f"Best time to visit {city_name} is in spring.",
        "currency": "Check XE.com for latest rates"
    }
    return city

def print_trip_plan(city_data, days):
    print("\n==============================")
    print(f"  Trip Plan for {city_data['name']}")
    print("==============================")
    print(f"Duration: {days} days")
    if days <= 3:
        print("Type: Short trip — focus on top 3 attractions only")
    elif days <= 7:
        print("Type: Good trip — explore neighbourhoods too")
    else:
        print("Type: Long trip — go off the beaten path!")
    print("\n--- Recommended Activities ---")
    for i, activity in enumerate(city_data["activities"], 1):
        print(f"{i}. {activity}")
    print("\n--- Travel Tip ---")
    print(city_data["tip"])
    print("\n--- Currency ---")
    print(city_data["currency"])
    print("==============================\n")

city_name = input("Enter a city you want to visit: ")
days = int(input("How many days are you travelling? "))
city_data = get_city_info(city_name, days)
print_trip_plan(city_data, days)