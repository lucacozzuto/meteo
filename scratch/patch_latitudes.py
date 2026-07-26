import glob

files = [
    "plot_extreme_rain_europe.py",
    "plot_hot_days_europe.py",
    "plot_records_yearly.py",
    "plot_records_yearly_min_europe.py",
    "plot_tropical_nights_europe.py"
]

for f in files:
    with open(f, "r") as file:
        content = file.read()
    
    if "Barcelona" not in content:
        content = content.replace(
            "'Madrid': 40.4168,", 
            "'Madrid': 40.4168, 'Barcelona': 41.35,"
        )
        
        with open(f, "w") as file:
            file.write(content)
        print(f"Updated {f}")
    else:
        print(f"Already updated {f}")
