
## Data source project description

For training a CLIP model all the data needs to be converted to the same format.

As there are multiple sources with different formats, there are several steps that need to be done. 
This simple project aims to accomplish this. 

### Overview of sources

1. High level human games \
These are downloaded from https://www.ffothello.org/informatique/la-base-wthor/ . The games follow the somewhat archaic thor file format that contains a lot of details that are not needed for this work, \
i.e. analysis of tournaments or player games. All these games have either been downloaded converted to pgn (from:https://github.com/MartinMSPedersen/othello-games/) or converted with the same script

2. Online human games #1 \
Similar to chess there are online sites, to play, the main (largest, though still a few magnitudes smaller ) is www.eothello.com \
A kaggle dataset was created for analysis earlier, it is reused here as well. (source: https://www.kaggle.com/datasets/andrefpoliveira/othello-games) \
The dataset contains high quality human games (played by the top100 players only)

3. Online human games #2 \
Similar to eothello.com , www.liveothello.com is another online site, broadcasting mainly human championships. The downloaded games are taken from there and contain high level human games

4. Synthetic games \
For training OthelloGPT (https://arxiv.org/html/2306.09200v2) lots of generated games were used. \
The synthetic dataset was donwloaded from links from https://github.com/likenneth/othello_world


### Need for conversion

Unfortunately the different sources provide the games with different attributes (data points) as well as in different format. \
The widely different storage requires conversion and unification. The sources/ formats are as follows: 

| Source              | Format              |
|---------------------|---------------------|
| Human OTB           | thor / pgn          |
| Human eothello      | csv                 |
| Human champiobnship | pgn                 |
| Synthetic           | pickle / python obj |

As part of the data preparation, hence all these source games are converted into the same format

Used tools: 
- a customized version of https://github.com/PypayaTech/pypaya-pgn-parser for pgn parsing
- 

### Unified format and storage

A pandas dataframe is used for storing all the games. This is not the most condensed or fastest solution by far, however, it allows a convenient handling, including serialisation, conversion, filtering etc.  

The columns of the dataframe are the following:

| Column      | Description                           | Comment                                                                               |
|-------------|---------------------------------------|---------------------------------------------------------------------------------------|
| ID          | Generated GUID                        | Not necessary, but for later database storage it can become handy                     |
| Event       | Event where the game was played       | Mandatory - set for synthetic                                                         |
| Black       | Name of the black player              | Mandatory - set for synthetic                                                         |
| White       | Name of the white player              | Mandatory - set for synthetic                                                         |                  |                                                                   |
| Result      | Result of the game                    | Number of stones for each side, i.e. 31-33                                            |
| Year / Date | Date or year when the game was played | Optional                                                                              |
| Source      | Where the data is coming from         | Can be useful for filtering (i.e. exclude synthetic data for high quality human games |   
| Moves       | Moves of the games                    | The moves of the games as a string folloewing the pgn notation                        |
| Hash        | Hash from all the moves               | (not yet implemented) useful for quick game search as well as duplication checking    |

### Post conversion results

At the time of writing the data sources have the following game numbers