Practical Application Assignment 11.1: What Drives the Price of a Car?

Context:
 This application,  will explore a dataset from kaggle. The provided dataset contains information on 426K cars. The goal is to understand what factors make a car more or less expensive. As a result of analysis,  I will provide clear recommendations to my client -- a used car dealership -- as to what consumers value in a used car.
 
Business Understanding: 

Most used car dealerships only deal in cars only that are only so old. We will assume that dealers only want to have cars up 10 years old on the lot, and that it is currently 2022. 

Does where the car is sold effect its value? 

What features of the car most influence the price of car? 

How well those features describe the price of a car?

What kind of cars should a dealership carry? Dealerships are only so big, so there should be a best car to have.

Data understanding:
First step is just to look of the data fields and categories, to see what can be known, look at info() and describe().

carsraw.info() # We have 18 columns, 5 numeric and 13 categorical, there are lots of missing fields from the non-null counts.

carsraw.describe() # There is a car listed from 1900, not likely useful to the model.

carsraw.sample(5) # Take a random samples to sample examples of all the fields (to get a feel for the data.)

Look at lengths, and information in the categories to decide on the encoding type to use for model.

Look through the value counts of categories field and look for misspellings/strange data.

id: is just a number we don't need for the model so will drop it

region: Large amount of unique values, but this information is also include in the state field (further down), so will drop and not keep in the model.

price: The price field is our primary target for the model, there are some zeros, and NANs included in the file, will ditch those rows as the will not help with a model.

year: Is the model year of the car.

manufacture: will use one hot encoding on this category.

model: will have to binary encode this category because there are so many!

condition: will use one hot encoding on this category.

cylinders: will convert this to a numeric field.

fuel: will use one hot encoding on this category.

title_status: will use one hot encoding on this category.

transmission: will use one hot encoding on this category. Wow, mostly automatic.

VIN: for modeling we don't need the VIN number, so we will drop it.

drive: will use one hot encoding, for this category, strange that are no all-wheel drive cars.

size: will use one hot encoding on this category.

type: will use one hot encoding on this category.

paint_color: will use one hot encoding on this category.

state: 51? It includes Washington DC, which isn't a state yet, will use binary encoding for this category.
Look for one offs (not useful for modeling since there is only one), really old cars will tend to be this way since so few of them are still around.

There are some classic cars (more than 20 years old) in the data but not many.

The dataset doesn't have any information on availability, which in the last two years has become critical; since new cars have become unavailable the prices of used cars has shot up.

The dataset also doesn't tell which year the cars were sold in, just the year of the car.

Data Preparation:

We can drop VIN and ID because we don't care about them, Region and state are related, likely only need one of them, so let’s keep state and drop region since they cover mostly the same information, and region would have to be binary encoded to use.

cars = cars[cars.year >= (2022-10)] # Car must be from the last 10 years. Remove older cars.

cars = cars[cars.price > 0] # Price of cars can't be zero.

cars = cars[cars.odometer > 0] # Odometer can't be zero, garbage data (even new cars normally have a mile on them.)

cars.loc[cars["transmission"] == "electric", "cylinders"] = '20' # Fix NaN on electric cylinders so we don't drop it. Set to large number to less break model.

cars.loc[(cars["size"].isnull())&(cars["type"] == 'sedan'), "size"] = 'full-size' # Sedans will be considered full-size if data is missing.

cars.loc[(cars["size"].isnull())&(cars["type"] == 'coupe'), "size"] = 'mid-size' # Coupes will be considered mid-size if data is missing.

cars.loc[(cars["size"].isnull())&(cars["type"] == 'van'), "size"] = 'full-size' # Vans will be considered full-size if data is missing.

cars.loc[(cars["type"] == 'pickup'), "type"] = 'truck' # Pickups are just trucks. Removal of sub-type.

cars.loc[(cars["size"].isnull())&(cars["type"] == 'truck'), "size"] = 'full-size' # Trucks of unknown size are full-size.

cars.loc[(cars["size"].isnull())&(cars["type"] == 'SUV'), "size"] = 'full-size' # SUVs of unknown size are full-size.

cars["cylinders"] = cars["cylinders"].str.replace('other','1') # Convert cylinders into a number, 1=other.

cars["cylinders"] = cars["cylinders"].str.replace('cylinders',' ') # Convert cylinders into a number, 20=electric.

cars['cylinders']=cars['cylinders'].astype(float) # Convert to float.

counts = cars.model.value_counts()

cars = cars.loc[cars['model'].isin(counts.index[counts > 3])] # Remove super rare cars, most dealership won't ever see/get one used anyway and they'd have to be researched on case by case basis.

cars['yearsold'] =  2022-cars['year'] # Convert model year into how many years old the car is.

cars = cars.drop(['year'], axis=1) # We can drop year, we'll just use how old the car is.

cars.sample(5) # Resample after cleanup.

cars["size"].isnull().sum() # Still a lot of bad data even after cleanup. 

complete = complete.dropna() # Cleanup remaiming NaN fields.

complete.drop(complete[complete['manufacturer'] == "harley-davidson"].index, inplace = True) # Drop, one off row!

complete = complete[complete.price > 1000] # Price can't be too low (scrap yard car, normally sent somewhere to either scrapped or be rebuilt.)

complete = complete[complete.price < 100000] # Price can't be too high (elite car, out of normal buyers range.)
46652 record left so plenty of data left to model / test the model on.

The heatmap shows that price is moderately correlated positively with cylinders, and moderately negatively correlated with years old, and odometer. 

We should see those come up in permutation importance.

Modeling:
X is every column but price, and y is just price.

Data is split into a training and test sets with 75% training and 25% testing.

Columns were feed into a preprocessor, then used in models. The Preprocessor runs a standard scaler on the numerical columns, does a binary encoder on model and state columns, and one hot encoded the remaining columns.
Several different models were setup:

All of them use TransformedTargetRegressor as the base. Models one and two used Ridge as the regressor, three and four used RandomForestRegressor as the regressor. Models two and four used a transformer the QuantileTransformer. 

A GridSearchCVs, were used on models one and three to tune the hyperparameters, alpha (for Ridge) and n_estimators for (RandomForest)

The first model MAE was 2730.56, with a tuned alpha (1.0) then the MAE was 2731.39 on the test set.

The second model MAE was 2610.00, on the test set.

The third model MAE was 704.33, with a tuned n_estimators (200) then the MAE was 689.39 on the test set.

The fourth model MAE was 795.00, on the test set.

The tuned model3 was the best of the models, with the lowest MAE.
Let's look at permutation importance next.

From model3, permutation importance:
r2

    cylinders 0.460 +/- 0.009
    
    odometer 0.308 +/- 0.005
    
    yearsold 0.261 +/- 0.005
    
    fuel     0.132 +/- 0.002
    
    drive    0.113 +/- 0.003
    
    type     0.110 +/- 0.003
    
    manufacturer 0.068 +/- 0.002
    
    model    0.040 +/- 0.001
    
    condition 0.017 +/- 0.001
    
    title_status 0.016 +/- 0.001
    
    paint_color 0.010 +/- 0.001
    
    transmission 0.007 +/- 0.000
    
    state    0.004 +/- 0.000
    
    size     0.003 +/- 0.000
    
Findings: How many big the engine is (by how many cylinders it has) dominates how valuable a car is, followed by how many miles on are it, followed by how many years old it is, followed by fuel.

This matches well with the correlation between the numeric data and the price.

A final model (model6) is made using the tuned model3, with a slightly changed Preprocessor.

The preprocessor now only one hot encodes categorical columns, and standard scales the numeric columns.
We can columns state, size, transmission are dropped because they did not affect results much. Column model is also dropped because it was less important than manufacturer and generally was making the model hard to use.
Model6 is predicting slightly better than model3, but still in the same range, it is also a slightly larger training/test set with a MAE of 634.47 on the test set. It has a 91% accuracy score at predictions which is fairly good.

Findings: 
From model63, permutation importance:
r2
    cylinders 0.463 +/- 0.009
    
    odometer 0.321 +/- 0.005
    
    yearsold 0.275 +/- 0.005
    
    fuel     0.132 +/- 0.002
    
    type     0.122 +/- 0.003
    
    drive    0.116 +/- 0.003
    
    manufacturer 0.087 +/- 0.003
    
    condition 0.059 +/- 0.002
    
    paint_color 0.023 +/- 0.001
    
    title_status 0.019 +/- 0.001
    
How many big the engine is (by how many cylinders it has) dominates how valuable a car is, followed by how many miles on are it, followed by how many years old it is, followed by fuel, type, drive, manufacturer, condition and paint_color and title_status.

The rest of the feature results will talked about in the Deployment section.

Deployment:
What state you sell the car in doesn't change the price of the car.

The features that make cars worth more (listed most important first):

	Large Engines, with lots of cylinders 
	
	Newer (Model Year)
	
	Low mileage 
	
	Fuel Type (in the listed order):
	
		Diesel, Gasoline
		
	Type of car (in the listed order):
	
		Trucks, sedan, SUV
		
	Drive Type:
	
		Fwd, 4wd
		
	With Clean Titles
	
	From Ford, Nissan, Toyota, Chevrolet, Honda
	
	The car's condition (in listed order):
	
		good, like new, excellent. (So good is good enough.) 
		
	Paint Colors (in the listed order):
	
		White, Black, Grey, Silver
		
So stock your lot with; Recent, low mileage Trucks, (from Ford/Nissan/Toyota) with large diesel engines (V8s,V10s,V12s) in White/Black/Grey/Silver and those will fetch the largest selling prices.

Remember you stand to make a killing on used cars, until production/supply chains on new cars get back to normal, which still make take a year or more. Availably is your customers’ biggest problem right now, which has been slowly getting better. So charge more while you still can.

Next steps for a better analysis:

Get data on availability of cars (including new). How scare cars currently are can and will affect pricing.

Get date of sales, so trends in features can be observed.

Get cleaner data, no reason to have such mangled data on the sale of cars in 2022.

Break the model column into more columns. It contained more than one feature to it, it contained both the model name, and model features (like 4x4 or hatchback.)

Get a dataset with more features, we have a really limited set of car features, like ABS, adaptive cruise control and stuff like that.
