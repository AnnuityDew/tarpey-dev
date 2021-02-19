# import native Python packages
from datetime import date
from enum import Enum
import multiprocessing
from typing import Dict
import orjson
from time import perf_counter

# import third party packages
from fastapi import APIRouter, HTTPException, Depends, Path
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
import pandas
from odmantic import AIOEngine, Field, Model
import requests
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# import custom local stuff
from instance.config import FANTASY_DATA_KEY_FREE
from src.db.atlas import get_odm
from src.api.users import oauth2_scheme


ab_api = APIRouter(
    prefix="/autobracket",
    tags=["autobracket"],
)


class FantasyDataSeason(str, Enum):
    PRIORSEASON1 = "2020"
    CURRENTSEASON = "2021"


class PlayerSeason(Model):
    StatID: int = Field(primary_field=True)
    TeamID: int
    PlayerID: int
    SeasonType: int
    Season: str
    Name: str
    Team: str
    Position: str
    Games: int
    FantasyPoints: float
    Minutes: int
    FieldGoalsMade: int
    FieldGoalsAttempted: int
    FieldGoalsPercentage: float
    TwoPointersMade: int
    TwoPointersAttempted: int
    ThreePointersMade: int
    ThreePointersAttempted: int
    FreeThrowsMade: int
    FreeThrowsAttempted: int
    OffensiveRebounds: int
    DefensiveRebounds: int
    Rebounds: int
    Assists: int
    Steals: int
    BlockedShots: int
    Turnovers: int
    PersonalFouls: int
    Points: int
    FantasyPointsFanDuel: float
    FantasyPointsDraftKings: float
    two_attempt_chance: float
    two_chance: float
    three_chance: float
    ft_chance: float


class KenpomTeam(Model):
    Season: FantasyDataSeason
    Rk: int
    Team: str
    Conf: str
    Wins: int
    Losses: int
    AdjEM: float
    AdjO: float
    AdjD: float
    AdjT: float
    Luck: float
    OppAdjEM: float
    OppO: float
    OppD: float
    NCAdjEM: float


class SimulationRun(Model):
    game_summary: Dict
    team_box_score: Dict
    full_box_score: Dict


@ab_api.get("/stats/{season}/all", dependencies=[Depends(oauth2_scheme)])
async def get_season_players(
    season: FantasyDataSeason,
    client: AsyncIOMotorClient = Depends(get_odm),
):
    engine = AIOEngine(motor_client=client, database="autobracket")
    data = [
        player_season
        async for player_season in engine.find(
            PlayerSeason,
            (PlayerSeason.Season == season),
            sort=PlayerSeason.StatID,
        )
    ]

    if data:
        return data
    else:
        raise HTTPException(status_code=404, detail="No data found!")


@ab_api.get("/stats/{season}/{team}", dependencies=[Depends(oauth2_scheme)])
async def get_season_team_players(
    season: FantasyDataSeason,
    team: str,
    client: AsyncIOMotorClient = Depends(get_odm),
):
    engine = AIOEngine(motor_client=client, database="autobracket")
    data = [
        player_season
        async for player_season in engine.find(
            PlayerSeason,
            (PlayerSeason.Season == season) & (PlayerSeason.Team == team),
            sort=PlayerSeason.StatID,
        )
    ]

    if data:
        return data
    else:
        raise HTTPException(status_code=404, detail="No data found!")


@ab_api.get("/sim/{season}/kmeans")
async def k_means_players(
    season: FantasyDataSeason,
    client: AsyncIOMotorClient = Depends(get_odm),
):

    engine = AIOEngine(motor_client=client, database="autobracket")
    player_data = [
        player_season
        async for player_season in engine.find(
            PlayerSeason,
            (PlayerSeason.Season == season),
            sort=(PlayerSeason.Team, PlayerSeason.StatID),
        )
    ]

    player_df = pandas.DataFrame(
        [player_season.doc() for player_season in player_data]
    ).set_index(["Team", "PlayerID"])

    # calculate potential columns for clustering, drop others
    player_df["points_per_second"] = player_df["Points"] / player_df["Minutes"] / 60
    player_df["shots_per_second"] = (
        player_df["FieldGoalsAttempted"] / player_df["Minutes"] / 60
    )
    player_df["rebounds_per_second"] = player_df["Rebounds"] / player_df["Minutes"] / 60
    player_df["assists_per_second"] = player_df["Assists"] / player_df["Minutes"] / 60
    player_df["steals_per_second"] = player_df["Steals"] / player_df["Minutes"] / 60
    player_df["blocks_per_second"] = (
        player_df["BlockedShots"] / player_df["Minutes"] / 60
    )
    player_df["turnovers_per_second"] = (
        player_df["Turnovers"] / player_df["Minutes"] / 60
    )
    player_df["fouls_per_second"] = (
        player_df["PersonalFouls"] / player_df["Minutes"] / 60
    )

    # minutes distribution for histogram
    hist_data = player_df["Minutes"].values.tolist()

    # drop anyone that didn't play a minute
    player_df = player_df.loc[player_df["Minutes"] > 100]

    # work in progress, but these will be the columns to start with
    player_df = player_df[
        [
            "two_attempt_chance",
            "two_chance",
            "three_chance",
            "ft_chance",
            # "points_per_second",
            # "shots_per_second",
            # "rebounds_per_second",
            # "assists_per_second",
            # "steals_per_second",
            # "blocks_per_second",
            # "turnovers_per_second",
            # "fouls_per_second",
        ]
    ]

    # min-max normalization
    player_df = (player_df - player_df.min()) / (player_df.max() - player_df.min())

    # columns for the scatter plot (do this before adding labels to the data)
    scatter_cols = player_df.columns.tolist()

    # K-Means time! 10 pretty much looks like where the elbow tapers off,
    # when looking at the four "rate" variables.
    model = KMeans(n_clusters=10)
    model.fit(player_df)
    player_df["player_type"] = model.predict(player_df)

    # for now just display 1/10th of the data
    player_df = player_df.sample(
        frac=0.2,
        replace=False,
    )

    # remove outliers (these are probably folks with very few minutes anyway)
    player_df = player_df.loc[(np.abs(stats.zscore(player_df)) < 3).all(axis=1)]

    scatter_data = orjson.loads(player_df.to_json(orient="records"))

    return {
        "scatter_data": scatter_data,
        "scatter_columns": scatter_cols,
        "inertia": model.inertia_,
        "hist_data": hist_data,
    }


@ab_api.get(
    "/sim/{season}/{away_team}/{home_team}/{sample_size}",
    dependencies=[Depends(oauth2_scheme)],
)
async def full_game_simulation(
    season: FantasyDataSeason,
    away_team: str,
    home_team: str,
    sample_size: int = Path(..., gt=0, le=10),
    client: AsyncIOMotorClient = Depends(get_odm),
):
    # performance timer
    start_time = perf_counter()

    engine = AIOEngine(motor_client=client, database="autobracket")
    matchup_data = [
        player_season
        async for player_season in engine.find(
            PlayerSeason,
            (PlayerSeason.Season == season)
            & ((PlayerSeason.Team == away_team) | (PlayerSeason.Team == home_team)),
            sort=(PlayerSeason.Team, PlayerSeason.StatID),
        )
    ]

    # create a dataframe representing one simulation
    matchup_df = pandas.DataFrame(
        [player_season.doc() for player_season in matchup_data]
    )
    # create an Away and Home field for identification in the simulation
    matchup_df["designation"] = "home"
    matchup_df.loc[matchup_df["Team"] == away_team, "designation"] = "away"

    # create a list of matchup dfs representing multiple simulations
    if False:
        cores_to_use = multiprocessing.cpu_count()
        simulations = [matchup_df.copy() for x in range(sample_size)]

        with multiprocessing.Pool(processes=cores_to_use) as p:
            results = p.map(run_simulation, simulations)
            # clean up
            p.close()
            p.join()
    else:
        # just do one run
        results = run_simulation(matchup_df, sample_size)

    sim_time = perf_counter()

    # write results to MongoDB
    await engine.save_all([SimulationRun(**doc) for doc in results])

    db_time = perf_counter()

    return {
        "sim_time": (sim_time - start_time),
        "db_time": (db_time - sim_time),
        "simulations": sample_size,
        "results": results,
    }


def run_simulation(matchup_df, sample_size):
    # sort df by designation and playerID to guarantee order for later operations
    matchup_df.sort_values(by=["designation", "PlayerID"], inplace=True)

    # home and away dict
    home_away_dict = dict(matchup_df.groupby(["designation", "Team"]).size().index)
    # new columns for simulated game stats
    sim_columns = [
        "sim_seconds",
        "sim_two_pointers_made",
        "sim_two_pointers_attempted",
        "sim_three_pointers_made",
        "sim_three_pointers_attempted",
        "sim_free_throws_made",
        "sim_free_throws_attempted",
        "sim_offensive_rebounds",
        "sim_defensive_rebounds",
        "sim_assists",
        "sim_steals",
        "sim_blocks",
        "sim_turnovers",
        "sim_fouls",
    ]
    for column in sim_columns:
        matchup_df[column] = 0

    # minutes for each player, divided by total minutes played for each team
    player_minute_totals = matchup_df.groupby(["Team", "PlayerID"]).agg(
        {"Minutes": "sum"}
    )
    team_minute_totals = matchup_df.groupby(["Team"]).agg({"Minutes": "sum"})
    player_time_share = player_minute_totals.div(
        team_minute_totals, level="Team"
    ).rename(columns={"Minutes": "minute_dist"})
    matchup_df = matchup_df.merge(
        player_time_share,
        left_on=["Team", "PlayerID"],
        right_index=True,
        how="left",
    )
    # minute weights will be used in the later step when we're selecting who's on the floor
    away_minute_weights = matchup_df.loc[
        matchup_df.designation == "away", ["PlayerID", "minute_dist"]
    ].to_numpy()
    home_minute_weights = matchup_df.loc[
        matchup_df.designation == "home", ["PlayerID", "minute_dist"]
    ].to_numpy()

    # new numpy random number generator
    rng = np.random.default_rng()

    # determine first possession in each game (simple 50/50 for now)
    matchup_list = team_minute_totals.index.to_list()
    possession_flags = rng.integers(2, size=sample_size)

    # game clock array, shot clock reset array, initialize possession length array,
    # possession counter for each team for each simulation
    time_remaining = np.array([60.0 * 40 for x in range(sample_size)])
    shot_clock_reset = np.ones(sample_size)
    possession_length = np.zeros(sample_size)
    total_possessions = np.zeros(sample_size)
    tempo_factor = 1 / 1

    # need to think about this more, but for now avg. possession length 15 as a normal
    possession_length_mean = 15  # 2400 / (Offensive Tempo + Defensive Tempo)
    possession_length_stdev = 4

    # set index that will be the basis for updating box score.
    matchup_df.set_index(["Team", "PlayerID"], inplace=True)

    # expand the dataframe into X number of simulations. (prepend the index)
    # https://stackoverflow.com/questions/14744068/prepend-a-level-to-a-pandas-multiindex
    matchup_df = pandas.concat(
        [matchup_df.copy() for x in range(sample_size)],
        keys=[x for x in range(sample_size)],
        names=["simulation"],
    )

    # loop continues while any game is still ongoing
    while max(time_remaining) >= 0:
        # who has the ball in each game?
        offensive_teams = [matchup_list[flag] for flag in possession_flags]
        defensive_teams = [matchup_list[1 - flag] for flag in possession_flags]

        # split df into games with time remaining and games with no time remaining
        ongoing_games = [sim for sim, value in enumerate(time_remaining) if value > 0]
        games_to_resolve = [
            sim for sim, value in enumerate(time_remaining) if value == 0
        ]
        ongoing_games_df = matchup_df.loc[ongoing_games]
        games_to_resolve_df = matchup_df.loc[games_to_resolve]

        # games to resolve might be over. calculate score and see if we need OT
        games_to_resolve_df["sim_points"] = (
            games_to_resolve_df["sim_free_throws_made"]
            + (games_to_resolve_df["sim_two_pointers_made"] * 2)
            + (games_to_resolve_df["sim_three_pointers_made"] * 3)
        )
        # aggregate team box score
        team_scores_df = games_to_resolve_df.groupby(level=[0, 1]).agg(
            {"sim_points": "sum"}
        )

        # compare score for each game to resolve. if tied, start overtime for that game
        for sim in games_to_resolve:
            if (
                team_scores_df.loc[(sim, offensive_teams[sim])][0]
                == team_scores_df.loc[(sim, defensive_teams[sim])][0]
            ):
                # start a 5 minute overtime!
                time_remaining[sim] = 60 * 5

        # this is where a check will go to end the loop if every game is resolved.
        print("this code is currently missing! need to check for end of all games.")

        # if there was a shot clock reset, this will add a possession to that particular game
        total_possessions += shot_clock_reset

        # if there was a shot clock reset, we want to use the value from this array of fresh
        # random numbers from the normal distribution. otherwise, use a squished distribution
        # based on the previous possession's length.
        # we definitely need to pull in some sort of tempo per team here,
        # but for now let's aim for a mean of 140 possessions per game.
        fresh_possession_length = rng.normal(
            loc=possession_length_mean, scale=possession_length_stdev, size=sample_size
        )
        recycled_possession_length = rng.normal(
            loc=possession_length_mean * (30 - possession_length / 30),
            scale=possession_length_stdev * (30 - possession_length / 30),
            size=sample_size,
        )
        # determine whether or not we should use the fresh possession or recycled in each game.
        # we can do this by multiplying by the shot_clock_reset_array (or its inverse)
        fresh_possession_length *= shot_clock_reset
        recycled_possession_length *= 1 - shot_clock_reset

        # now add the two together to get the new possession length for each game
        # will either look like x+0 or 0+x for each row
        possession_length = fresh_possession_length + recycled_possession_length

        # expect another shot clock reset in the next possession by default
        # (we overwrite this once an event occurs, if necessary)
        shot_clock_reset = np.ones(sample_size)

        # pick 10 players for the current possession based on average time share
        # pandas has a bug so we're doing this with numpy now.
        on_floor_all_sims = np.array(
            [
                np.concatenate(
                    (
                        np.isin(
                            away_minute_weights[:, 0],
                            rng.choice(
                                away_minute_weights[:, 0],
                                size=5,
                                replace=False,
                                p=away_minute_weights[:, 1],
                            ),
                        )
                        * 1,
                        np.isin(
                            home_minute_weights[:, 0],
                            rng.choice(
                                home_minute_weights[:, 0],
                                size=5,
                                replace=False,
                                p=home_minute_weights[:, 1],
                            ),
                        )
                        * 1,
                    ),
                )
                for x in range(sample_size)
            ]
        ).flatten()
        matchup_df["player_on_floor"] = on_floor_all_sims
        # copy here to avoid a later settingwithcopywarning
        on_floor_df = matchup_df.loc[matchup_df.player_on_floor == 1].copy()
        # on_floor_df = matchup_df.groupby(level=[0, 1]).sample(
        #     n=5, replace=False, weights=matchup_df.minute_dist.to_list()
        # )

        # add the possession length to the time played for each individual on the floor and update.
        # numpy array is expanded 10x so each player of the 10 on the floor can get their time
        on_floor_df["sim_seconds"] += np.repeat(possession_length, 10)
        matchup_df.update(on_floor_df)

        # now, based on the 10 players on the floor, calculate probability of each event.
        # first, a steal check happens here. use steals per second over the season.
        # improvement: factor in the opponent's turnover statistics here.
        # steals per second times possession length to get the steal chance for this possession
        # times 2 assumes each team has the ball for about half the game. this effectively
        # converts steals per both teams' possession to steals per defensive possession (since
        # you can't get a steal while you're on offense!)
        # i think this is where we would put a tempo factor...
        steal_probs = steal_distribution(
            possession_length, defensive_teams, on_floor_df, tempo_factor
        )

        # we also need turnover probabilities here
        turnover_probs = turnover_distribution(
            possession_length, offensive_teams, on_floor_df, tempo_factor
        )

        # the steal/turnover check! we're modeling them as independent.
        # (right now it's possible that a turnover in a given possession
        # will always be a steal, if turnover_chance is less than steal_chance.)
        # we'll also not use prior probabilities in the turnover logic
        # for now, for this reason.
        steal_turnover_success = rng.random(size=10)
        team_steal_chances = (
            1 - steal_probs.groupby(level=0).prod()["no_steal_chance"].to_numpy()
        )
        team_turnover_chances = (
            1 - turnover_probs.groupby(level=0).prod()["no_turnover_chance"].to_numpy()
        )

        # if there's a successful steal, credit the steal and turnover, then flip possession.
        # games with steals don't do anything else until the loop restarts for a new possession.
        successful_steals = steal_turnover_success < team_steal_chances
        steal_games = [sim for sim, value in enumerate(successful_steals) if value]
        successful_turnovers = steal_turnover_success < team_turnover_chances
        turnover_games = [sim for sim, value in enumerate(successful_turnovers) if value]

        # who got the steal in each game that had a steal?
        # pandas really needs to fix their groupby sampling...
        # let's try to reproduce the issue later with this commit, which worked:
        # https://github.com/AnnuityDew/tarpeydev/blob/ba344c7b29f3385caf5c964f10accc43ab600bd3/src/api/autobracket.py#L414
        # hypothesis is that it's because the resulting index is no longer unique.
        # but that can't be right because i duplicated the first index level
        # and tried to group from that as well.
        # for now we'll just have to keep sampling with numpy.
        # i'm really starting to wonder if it's because weights don't sum to one...
        steal_games_df = steal_probs.loc[steal_games]
        steal_games_numpy = steal_games_df.reset_index()[
            ["simulation", "Team", "PlayerID", "steal_chance"]
        ].to_numpy()
        turnover_games_df = turnover_probs.loc[turnover_games]
        turnover_games_numpy = turnover_games_df.reset_index()[
            ["simulation", "Team", "PlayerID", "turnover_chance"]
        ].to_numpy()

        # steal array has a 1 for the player in each steal game that got the steal
        # APPLY THIS EVENT SAMPLER TO THE OTHER EVENTS.
        steal_array = event_sampler(rng, steal_games_df, steal_games_numpy)
        turnover_array = event_sampler(rng, turnover_games_df, turnover_games_numpy)

        steal_games_df["sim_steals"] += steal_array
        turnover_games_df["sim_turnovers"] += turnover_array
        matchup_df.update(steal_games_df)
        matchup_df.update(turnover_games_df)

        print("then, need to figure out the time remaining, possession length, and possession flag arrays.")

        # for games with a steal/turnover, give ball to other team and
        # flag them as inactive until the loop resets.
        # how do we do this? ndenumerate maybe?
        # answer might be a second row in the possession flags array that populates
        # based on the steal/turnover arrays. then, can do np.where to change
        # the first row (1-first row) based on whether there's a true in the second row.
        # third row could be the indicator for "don't do anything more with this game
        # until the loop resets"
        possession_flag = 1 - possession_flag

        return possession_flag

        # if there's a turnover, credit the turnover, then flip possession and restart loop!
        if steal_turnover_success < team_turnover_chances:
            # who committed the turnover?
            turnover_player = turnover_probs.sample(
                n=1, weights=turnover_probs.turnover_chance
            )
            turnover_player["sim_turnovers"] = turnover_player["sim_turnovers"] + 1
            # update box score, change clock, give ball to other team, reset loop
            matchup_df.update(turnover_player)
            time_remaining -= possession_length
            possession_flag = 1 - possession_flag
            continue

        # time to model shot attempts. if there's no steal or turnover,
        # a shot is the only other outcome, so we can simply model who's
        # gonna take it and what kind of shot it will be.
        shooting_player = identify_shooter(offensive_team, on_floor_df)

        # if a defensive player blocks, 50/50 chance to be a rebound.
        # using blocks per second over the season.
        # we're either crediting miss+block, or miss+block+rebound.
        # we also need to update and provide the given probability of
        # making it this far
        given_probabilities = 1 - team_turnover_chances
        block_probs = block_distribution(
            possession_length,
            defensive_team,
            on_floor_df,
            tempo_factor,
            given_probability,
        )

        # block check!
        block_success = rng.random()
        team_block_chance = 1 - block_probs.no_block_chance.product()

        # the shot type check!
        two_or_three = rng.random()

        if block_success < team_block_chance:
            # who got the block?
            blocking_player = block_probs.sample(n=1, weights=block_probs.block_chance)
            blocking_player["sim_blocks"] = blocking_player["sim_blocks"] + 1
            matchup_df.update(blocking_player)

            if two_or_three < shooting_player.two_attempt_chance.values[0]:
                # credit the shooter with a 2pt attempt
                shooting_player["sim_two_pointers_attempted"] = (
                    shooting_player["sim_two_pointers_attempted"] + 1
                )
            else:
                # credit the shooter with a 3pt attempt
                shooting_player["sim_three_pointers_attempted"] = (
                    shooting_player["sim_three_pointers_attempted"] + 1
                )
            matchup_df.update(shooting_player)

            # block out of bounds check! this is 50/50 for now.
            block_oob_check = rng.integers(2, size=1)[0]

            if block_oob_check == 1:
                # no change of possession, don't reset shot clock
                time_remaining -= possession_length
                shot_clock_reset = not shot_clock_reset
                continue
            else:
                # rebound logic
                (
                    offensive_rebound_probs,
                    defensive_rebound_probs,
                    off_reb_chance,
                ) = rebound_distribution(offensive_team, defensive_team, on_floor_df)

                # rebound type check!
                off_reb_success = rng.random()

                if off_reb_success < off_reb_chance:
                    # who got the rebound?
                    rebounding_player = offensive_rebound_probs.sample(
                        n=1, weights=offensive_rebound_probs.off_reb_pdf
                    )
                    rebounding_player["sim_offensive_rebounds"] = (
                        rebounding_player["sim_offensive_rebounds"] + 1
                    )
                    matchup_df.update(rebounding_player)

                    # no change of possession, don't reset shot clock
                    time_remaining -= possession_length
                    shot_clock_reset = not shot_clock_reset
                    continue
                else:
                    # who got the rebound?
                    rebounding_player = defensive_rebound_probs.sample(
                        n=1, weights=defensive_rebound_probs.def_reb_pdf
                    )
                    rebounding_player["sim_defensive_rebounds"] = (
                        rebounding_player["sim_defensive_rebounds"] + 1
                    )
                    matchup_df.update(rebounding_player)

                    # update clock, change of possession, reset loop
                    time_remaining -= possession_length
                    possession_flag = 1 - possession_flag
                    continue

        # we need to prep for assist and foul calculation here. will involve Bayes,
        # so we need the components for probability of a successful shot
        # in this possession (1 minus everything that had to be dodged up
        # to this point). works as decrementing essentially because
        # the sequence of events isn't truly independent (i.e. if a turnover
        # happens in this model, a block will not because the loop restarts)
        given_probability = given_probability * (1 - team_block_chance)

        # if we've made it this far, the shot was not blocked.
        # but did it go in? and was there a foul?
        foul_probs = foul_distribution(
            possession_length,
            defensive_team,
            on_floor_df,
            tempo_factor,
            given_probability,
        )

        # defensive foul check! (potential improvement, offensive fouls and
        # non-shooting fouls)
        team_foul_chance = 1 - foul_probs.no_foul_chance.product()
        foul_occurred = rng.random()

        # non shooting foul check! this is 50/50 for now.
        if foul_occurred < team_foul_chance:
            non_shooting_foul_check = rng.integers(2, size=1)[0]

            if non_shooting_foul_check == 1:
                # no change of possession. credit a non-shooting foul.
                # make a new possession
                fouling_player = foul_probs.sample(n=1, weights=foul_probs.foul_chance)
                fouling_player["sim_fouls"] = fouling_player["sim_fouls"] + 1
                matchup_df.update(fouling_player)

                time_remaining -= possession_length
                shot_clock_reset = not shot_clock_reset
                continue

        if two_or_three < shooting_player.two_attempt_chance.values[0]:
            # check two point probability
            two_success = rng.random()
            made_two_chance = shooting_player.two_chance.values[0]

            if two_success < made_two_chance:
                # credit the shooter with a 2pt attempt and make
                shooting_player["sim_two_pointers_attempted"] = (
                    shooting_player["sim_two_pointers_attempted"] + 1
                )
                shooting_player["sim_two_pointers_made"] = (
                    shooting_player["sim_two_pointers_made"] + 1
                )
                if foul_occurred < team_foul_chance:
                    # and-1! first, credit the foul
                    fouling_player = foul_probs.sample(
                        n=1, weights=foul_probs.foul_chance
                    )
                    fouling_player["sim_fouls"] = fouling_player["sim_fouls"] + 1
                    matchup_df.update(fouling_player)

                    # credit the free throw attempt
                    shooting_player["sim_free_throws_attempted"] = (
                        shooting_player["sim_free_throws_attempted"] + 1
                    )

                    # now simulate free throw success
                    ft_success_and_1 = rng.random()
                    if ft_success_and_1 < shooting_player.ft_chance.values[0]:
                        shooting_player["sim_free_throws_made"] = (
                            shooting_player["sim_free_throws_made"] + 1
                        )

                matchup_df.update(shooting_player)

                # assist logic after a made shot. who assisted it if anyone?
                given_probability = given_probability * made_two_chance
                assist_probs = assist_distribution(
                    possession_length,
                    offensive_team,
                    on_floor_df,
                    tempo_factor,
                    given_probability,
                )

                # assist check!
                assist_success = rng.random()
                team_assist_chance = 1 - assist_probs.no_assist_chance.product()

                if assist_success < team_assist_chance:
                    # who got the assist?
                    assisting_player = assist_probs.sample(
                        n=1, weights=assist_probs.assist_chance
                    )
                    assisting_player["sim_assists"] = (
                        assisting_player["sim_assists"] + 1
                    )
                    matchup_df.update(assisting_player)

                # change clock, give ball to other team, reset loop
                time_remaining -= possession_length
                possession_flag = 1 - possession_flag
                continue

            else:
                if foul_occurred < team_foul_chance:
                    # if fouled, two FTs, and shooter is NOT credited with a missed shot
                    fouling_player = foul_probs.sample(
                        n=1, weights=foul_probs.foul_chance
                    )
                    fouling_player["sim_fouls"] = fouling_player["sim_fouls"] + 1
                    matchup_df.update(fouling_player)

                    # credit the free throw attempts
                    shooting_player["sim_free_throws_attempted"] = (
                        shooting_player["sim_free_throws_attempted"] + 2
                    )

                    # now simulate free throw success
                    ft_success_shot_1 = rng.random()
                    ft_success_shot_2 = rng.random()
                    for ft_success in [ft_success_shot_1, ft_success_shot_2]:
                        if ft_success < shooting_player.ft_chance.values[0]:
                            shooting_player["sim_free_throws_made"] = (
                                shooting_player["sim_free_throws_made"] + 1
                            )
                    matchup_df.update(shooting_player)
                    # if the last free throw is made, turn the ball over.
                    if ft_success_shot_2 < shooting_player.ft_chance.values[0]:
                        time_remaining -= possession_length
                        possession_flag = 1 - possession_flag
                        continue
                    # if the last free throw is missed, it's a rebound situation!
                    else:
                        # who gets the rebound?
                        (
                            offensive_rebound_probs,
                            defensive_rebound_probs,
                            off_reb_chance,
                        ) = rebound_distribution(
                            offensive_team, defensive_team, on_floor_df
                        )

                        # rebound type check!
                        off_reb_success = rng.random()

                        if off_reb_success < off_reb_chance:
                            # who got the rebound?
                            rebounding_player = offensive_rebound_probs.sample(
                                n=1, weights=offensive_rebound_probs.off_reb_pdf
                            )
                            rebounding_player["sim_offensive_rebounds"] = (
                                rebounding_player["sim_offensive_rebounds"] + 1
                            )
                            matchup_df.update(rebounding_player)

                            # no change of possession, don't reset shot clock
                            time_remaining -= possession_length
                            shot_clock_reset = not shot_clock_reset
                            continue
                        else:
                            # who got the rebound?
                            rebounding_player = defensive_rebound_probs.sample(
                                n=1, weights=defensive_rebound_probs.def_reb_pdf
                            )
                            rebounding_player["sim_defensive_rebounds"] = (
                                rebounding_player["sim_defensive_rebounds"] + 1
                            )
                            matchup_df.update(rebounding_player)

                            # update clock, change of possession, reset loop
                            time_remaining -= possession_length
                            possession_flag = 1 - possession_flag
                            continue
                else:
                    shooting_player["sim_two_pointers_attempted"] = (
                        shooting_player["sim_two_pointers_attempted"] + 1
                    )
                    matchup_df.update(shooting_player)

                    # who gets the rebound?
                    (
                        offensive_rebound_probs,
                        defensive_rebound_probs,
                        off_reb_chance,
                    ) = rebound_distribution(
                        offensive_team, defensive_team, on_floor_df
                    )

                    # rebound type check!
                    off_reb_success = rng.random()

                    if off_reb_success < off_reb_chance:
                        # who got the rebound?
                        rebounding_player = offensive_rebound_probs.sample(
                            n=1, weights=offensive_rebound_probs.off_reb_pdf
                        )
                        rebounding_player["sim_offensive_rebounds"] = (
                            rebounding_player["sim_offensive_rebounds"] + 1
                        )
                        matchup_df.update(rebounding_player)

                        # no change of possession, don't reset shot clock
                        time_remaining -= possession_length
                        shot_clock_reset = not shot_clock_reset
                        continue
                    else:
                        # who got the rebound?
                        rebounding_player = defensive_rebound_probs.sample(
                            n=1, weights=defensive_rebound_probs.def_reb_pdf
                        )
                        rebounding_player["sim_defensive_rebounds"] = (
                            rebounding_player["sim_defensive_rebounds"] + 1
                        )
                        matchup_df.update(rebounding_player)

                        # update clock, change of possession, reset loop
                        time_remaining -= possession_length
                        possession_flag = 1 - possession_flag
                        continue

        else:
            # check three point probability
            three_success = rng.random()
            made_three_chance = shooting_player.three_chance.values[0]

            if three_success < shooting_player.three_chance.values[0]:
                # credit the shooter with a 2pt attempt and make
                shooting_player["sim_three_pointers_attempted"] = (
                    shooting_player["sim_three_pointers_attempted"] + 1
                )
                shooting_player["sim_three_pointers_made"] = (
                    shooting_player["sim_three_pointers_made"] + 1
                )
                if foul_occurred < team_foul_chance:
                    # and-1! first, credit the foul
                    fouling_player = foul_probs.sample(
                        n=1, weights=foul_probs.foul_chance
                    )
                    fouling_player["sim_fouls"] = fouling_player["sim_fouls"] + 1
                    matchup_df.update(fouling_player)

                    # credit the free throw attempt
                    shooting_player["sim_free_throws_attempted"] = (
                        shooting_player["sim_free_throws_attempted"] + 1
                    )

                    # now simulate free throw success
                    ft_success_and_1 = rng.random()
                    if ft_success_and_1 < shooting_player.ft_chance.values[0]:
                        shooting_player["sim_free_throws_made"] = (
                            shooting_player["sim_free_throws_made"] + 1
                        )

                matchup_df.update(shooting_player)

                # assist logic after a made shot. who assisted it if anyone?
                given_probability = given_probability * made_three_chance
                assist_probs = assist_distribution(
                    possession_length,
                    offensive_team,
                    on_floor_df,
                    tempo_factor,
                    given_probability,
                )

                # assist check!
                assist_success = rng.random()
                team_assist_chance = 1 - assist_probs.no_assist_chance.product()

                if assist_success < team_assist_chance:
                    # who got the assist?
                    assisting_player = assist_probs.sample(
                        n=1, weights=assist_probs.assist_chance
                    )
                    assisting_player["sim_assists"] = (
                        assisting_player["sim_assists"] + 1
                    )
                    matchup_df.update(assisting_player)

                # change clock, give ball to other team, reset loop
                time_remaining -= possession_length
                possession_flag = 1 - possession_flag
                continue

            else:
                if foul_occurred < team_foul_chance:
                    # if fouled, three FTs, and shooter is NOT credited with a missed shot
                    fouling_player = foul_probs.sample(
                        n=1, weights=foul_probs.foul_chance
                    )
                    fouling_player["sim_fouls"] = fouling_player["sim_fouls"] + 1
                    matchup_df.update(fouling_player)

                    # credit the free throw attempts
                    shooting_player["sim_free_throws_attempted"] = (
                        shooting_player["sim_free_throws_attempted"] + 3
                    )

                    # now simulate free throw success
                    ft_success_shot_1 = rng.random()
                    ft_success_shot_2 = rng.random()
                    for ft_success in [ft_success_shot_1, ft_success_shot_2]:
                        if ft_success < shooting_player.ft_chance.values[0]:
                            shooting_player["sim_free_throws_made"] = (
                                shooting_player["sim_free_throws_made"] + 1
                            )
                    matchup_df.update(shooting_player)
                    # if the last free throw is made, turn the ball over.
                    if ft_success_shot_2 < shooting_player.ft_chance.values[0]:
                        time_remaining -= possession_length
                        possession_flag = 1 - possession_flag
                        continue
                    # if the last free throw is missed, it's a rebound situation!
                    else:
                        # who gets the rebound?
                        (
                            offensive_rebound_probs,
                            defensive_rebound_probs,
                            off_reb_chance,
                        ) = rebound_distribution(
                            offensive_team, defensive_team, on_floor_df
                        )

                        # rebound type check!
                        off_reb_success = rng.random()

                        if off_reb_success < off_reb_chance:
                            # who got the rebound?
                            rebounding_player = offensive_rebound_probs.sample(
                                n=1, weights=offensive_rebound_probs.off_reb_pdf
                            )
                            rebounding_player["sim_offensive_rebounds"] = (
                                rebounding_player["sim_offensive_rebounds"] + 1
                            )
                            matchup_df.update(rebounding_player)

                            # no change of possession, don't reset shot clock
                            time_remaining -= possession_length
                            shot_clock_reset = not shot_clock_reset
                            continue
                        else:
                            # who got the rebound?
                            rebounding_player = defensive_rebound_probs.sample(
                                n=1, weights=defensive_rebound_probs.def_reb_pdf
                            )
                            rebounding_player["sim_defensive_rebounds"] = (
                                rebounding_player["sim_defensive_rebounds"] + 1
                            )
                            matchup_df.update(rebounding_player)

                            # update clock, change of possession, reset loop
                            time_remaining -= possession_length
                            possession_flag = 1 - possession_flag
                            continue
                else:
                    shooting_player["sim_three_pointers_attempted"] = (
                        shooting_player["sim_three_pointers_attempted"] + 1
                    )
                    matchup_df.update(shooting_player)

                    # who gets the rebound?
                    (
                        offensive_rebound_probs,
                        defensive_rebound_probs,
                        off_reb_chance,
                    ) = rebound_distribution(
                        offensive_team, defensive_team, on_floor_df
                    )

                    # rebound type check!
                    off_reb_success = rng.random()

                    if off_reb_success < off_reb_chance:
                        # who got the rebound?
                        rebounding_player = offensive_rebound_probs.sample(
                            n=1, weights=offensive_rebound_probs.off_reb_pdf
                        )
                        rebounding_player["sim_offensive_rebounds"] = (
                            rebounding_player["sim_offensive_rebounds"] + 1
                        )
                        matchup_df.update(rebounding_player)

                        # no change of possession, don't reset shot clock
                        time_remaining -= possession_length
                        shot_clock_reset = not shot_clock_reset
                        continue
                    else:
                        # who got the rebound?
                        rebounding_player = defensive_rebound_probs.sample(
                            n=1, weights=defensive_rebound_probs.def_reb_pdf
                        )
                        rebounding_player["sim_defensive_rebounds"] = (
                            rebounding_player["sim_defensive_rebounds"] + 1
                        )
                        matchup_df.update(rebounding_player)

                        # update clock, change of possession, reset loop
                        time_remaining -= possession_length
                        possession_flag = 1 - possession_flag
                        continue

        # update clocks in all games, then restart the loop
        time_remaining -= possession_length
        continue

    # that's the end of the loop.
    # time to set up the box scores!
    box_score_df = matchup_df[
        [
            "Name",
            "Position",
            "sim_seconds",
            "sim_two_pointers_made",
            "sim_two_pointers_attempted",
            "sim_three_pointers_made",
            "sim_three_pointers_attempted",
            "sim_free_throws_made",
            "sim_free_throws_attempted",
            "sim_offensive_rebounds",
            "sim_defensive_rebounds",
            "sim_assists",
            "sim_steals",
            "sim_blocks",
            "sim_turnovers",
            "sim_fouls",
            "sim_points",
        ]
    ]

    # calculate totals and downcast to ints
    box_score_df = box_score_df.assign(
        sim_minutes=lambda x: x.sim_seconds / 60
    ).convert_dtypes()

    # aggregate team box score and downcast to ints
    team_box_score_df = box_score_df.groupby(level=0).sum().convert_dtypes()

    full_box_score_json = orjson.loads(box_score_df.to_json(orient="index"))
    team_box_score_json = orjson.loads(team_box_score_df.to_json(orient="index"))
    margin = int(
        team_box_score_df.loc[home_away_dict["home"], "sim_points"]
        - team_box_score_df.loc[home_away_dict["away"], "sim_points"]
    )

    return {
        "game_summary": {
            "away_team": home_away_dict["away"],
            "home_team": home_away_dict["home"],
            "home_margin": margin,
            "total_possessions": total_possessions,
        },
        "team_box_score": team_box_score_json,
        "full_box_score": full_box_score_json,
    }


def event_sampler(rng, games_df, games_numpy):
    event_array = np.array(
            [
                np.isin(
                    games_numpy[x : x + 5, 2],
                    rng.choice(
                        games_numpy[x : x + 5, 2],
                        size=1,
                        replace=False,
                        p=(
                            games_numpy[0:5, 3] / games_numpy[0:5, 3].sum()
                        ).astype(float),
                    ),
                )
                for x in range(0, len(games_df), 5)
            ]
        ).flatten() * 1
    return event_array


def assist_distribution(
    possession_length, offensive_team, on_floor_df, tempo_factor, successful_shot_prob
):
    assist_fields = [
        "Assists",
        "Minutes",
        "sim_assists",
    ]
    assist_probs = on_floor_df.loc[[offensive_team], assist_fields]
    # alternative...let's try truly modeling as an exponential distribution.
    # first calculate rate parameter (per game estimate)
    assist_probs["assist_chance_exp_theta"] = 1 / (
        (2 * tempo_factor) * assist_probs["Assists"] / assist_probs["Minutes"] / 60
    )
    # now use rate parameter to calculate CDF per player, per possession.
    # x = the percentage of the team's gametime that has elapsed
    # numpy array is expanded 5x so each player of the 5 on this side can get their time
    assist_probs["no_assist_chance_prior"] = np.e ** (
        -1 * np.repeat(possession_length, 5) / assist_probs["assist_chance_exp_theta"]
    )
    # we could also randomly sample an exponential for each player using numpy
    # and see if it's less than the number of possession_seconds. might be
    # interesting to see how this changes the simulation.
    assist_probs["assist_chance_prior"] = 1 - assist_probs["no_assist_chance_prior"]

    # we need to do some bayes here to handle assists properly! everything
    # up to this point was modeled as independent, but assists can only happen
    # given a made shot. So we calculate P(assist|made shot) here.
    assist_probs["assist_chance"] = (
        assist_probs["assist_chance_prior"] / successful_shot_prob
    )
    assist_probs["no_assist_chance"] = 1 - assist_probs["assist_chance"]

    return assist_probs


def foul_distribution(
    possession_length, defensive_teams, on_floor_df, tempo_factor, attempted_shot_prob
):
    foul_fields = [
        "PersonalFouls",
        "Minutes",
        "sim_fouls",
    ]
    # dropping the PlayerID level allows us to slice out only the teams on defense in each sim.
    defensive_index = [(sim, team) for sim, team in enumerate(defensive_teams)]
    foul_probs = on_floor_df.loc[
        on_floor_df.index.droplevel("PlayerID").isin(defensive_index), foul_fields
    ]

    # the x2 factor is still here for fouls, even though you can foul on offense or defense.
    # but for now this model is assuming you can only foul on defense. we would scrap
    # this factor once we're modeling offensive fouls independently

    # let's try truly modeling as an exponential distribution.
    # first calculate rate parameter (1 / theta or beta), aka
    # (1 / average amount of seconds between two events)
    # factor of 2 * tempo_factor because you can only steal
    # when you're playing on defense!
    foul_probs["foul_chance_exp_theta"] = 1 / (
        (2 * tempo_factor) * foul_probs["PersonalFouls"] / foul_probs["Minutes"] / 60
    )
    # now use rate parameter to calculate CDF per player, per possession.
    # x = the percentage of the team's gametime that has elapsed
    # numpy array is expanded 5x so each player of the 5 on this side can get their time
    foul_probs["no_foul_chance_prior"] = np.e ** (
        -1 * np.repeat(possession_length, 5) / foul_probs["foul_chance_exp_theta"]
    )
    # we could also randomly sample an exponential for each player using numpy
    # and see if it's less than the number of possession_seconds. might be
    # interesting to see how this changes the simulation.
    foul_probs["foul_chance_prior"] = 1 - foul_probs["no_foul_chance_prior"]

    # we need to do some bayes here to handle fouls properly! everything
    # up to this point was modeled as independent, but fouls can only happen
    # given a made shot. So we calculate P(foul|made shot) here.
    foul_probs["foul_chance"] = foul_probs["foul_chance_prior"] / attempted_shot_prob
    foul_probs["no_foul_chance"] = 1 - foul_probs["foul_chance"]

    return foul_probs


def block_distribution(
    possession_length, defensive_teams, on_floor_df, tempo_factor, no_turnover_prob
):
    block_fields = ["BlockedShots", "Minutes", "sim_blocks"]
    # dropping the PlayerID level allows us to slice out only the teams on defense in each sim.
    defensive_index = [(sim, team) for sim, team in enumerate(defensive_teams)]
    block_probs = on_floor_df.loc[
        on_floor_df.index.droplevel("PlayerID").isin(defensive_index), block_fields
    ]

    # let's try truly modeling as an exponential distribution.
    # first calculate rate parameter (1 / theta or beta), aka
    # (1 / average amount of seconds between two events)
    # factor of 2 * tempo_factor because you can only steal
    # when you're playing on defense!
    block_probs["block_chance_exp_theta"] = 1 / (
        (2 * tempo_factor) * block_probs["BlockedShots"] / block_probs["Minutes"] / 60
    )
    # now use rate parameter to calculate CDF per player, per possession.
    # x = the percentage of the team's gametime that has elapsed
    # numpy array is expanded 5x so each player of the 5 on this side can get their time
    block_probs["no_block_chance_prior"] = np.e ** (
        -1 * np.repeat(possession_length, 5) / block_probs["block_chance_exp_theta"]
    )
    # we could also randomly sample an exponential for each player using numpy
    # and see if it's less than the number of possession_seconds. might be
    # interesting to see how this changes the simulation.
    block_probs["block_chance_prior"] = 1 - block_probs["no_block_chance_prior"]

    # we need to do some bayes here to handle blocks properly! blocks can
    # only happen in our model given no turnover. so we calculate
    # P(block|no turnover) here.
    block_probs["block_chance"] = block_probs["block_chance_prior"] / no_turnover_prob
    block_probs["no_block_chance"] = 1 - block_probs["block_chance"]

    return block_probs


def identify_shooter(offensive_teams, on_floor_df):
    shot_fields = [
        "two_attempt_chance",
        "two_chance",
        "three_chance",
        "ft_chance",
        "TwoPointersAttempted",
        "TwoPointersMade",
        "ThreePointersAttempted",
        "ThreePointersMade",
        "FieldGoalsAttempted",
        "FieldGoalsMade",
        "FreeThrowsAttempted",
        "FreeThrowsMade",
        "sim_two_pointers_made",
        "sim_two_pointers_attempted",
        "sim_three_pointers_made",
        "sim_three_pointers_attempted",
        "sim_free_throws_made",
        "sim_free_throws_attempted",
    ]
    shot_probs = on_floor_df.loc[[offensive_team], shot_fields]
    team_shot_prob = shot_probs.groupby(level=0).sum()
    shot_share = shot_probs.div(team_shot_prob, level="Team")
    shot_probs["field_goal_pdf"] = shot_share["FieldGoalsAttempted"]

    # identify the shooter
    shooting_player = shot_probs.sample(n=1, weights=shot_probs.field_goal_pdf)

    return shooting_player


def steal_distribution(possession_length, defensive_teams, on_floor_df, tempo_factor):
    steal_fields = [
        "Steals",
        "Minutes",
        "sim_steals",
    ]
    # dropping the PlayerID level allows us to slice out only the teams on defense in each sim.
    defensive_index = [(sim, team) for sim, team in enumerate(defensive_teams)]
    steal_probs = on_floor_df.loc[
        on_floor_df.index.droplevel("PlayerID").isin(defensive_index), steal_fields
    ]
    # let's try truly modeling as an exponential distribution.
    # first calculate rate parameter (1 / theta or beta), aka
    # (1 / average amount of seconds between two events)
    # factor of 2 * tempo_factor because you can only steal
    # when you're playing on defense!
    steal_probs["steal_chance_exp_theta"] = 1 / (
        (2 * tempo_factor) * steal_probs["Steals"] / steal_probs["Minutes"] / 60
    )
    # now use rate parameter to calculate CDF per player, per possession.
    # x = the percentage of the team's gametime that has elapsed
    # numpy array is expanded 5x so each player of the 5 on this side can get their time
    steal_probs["no_steal_chance"] = np.e ** (
        -1 * np.repeat(possession_length, 5) / steal_probs["steal_chance_exp_theta"]
    )
    # we could also randomly sample an exponential for each player using numpy
    # and see if it's less than the number of possession_seconds. might be
    # interesting to see how this changes the simulation.
    steal_probs["steal_chance"] = 1 - steal_probs["no_steal_chance"]

    return steal_probs


def turnover_distribution(
    possession_length, offensive_teams, on_floor_df, tempo_factor
):
    turnover_fields = [
        "Turnovers",
        "Minutes",
        "sim_turnovers",
    ]
    # dropping the PlayerID level allows us to slice out only the teams on offensive in each sim.
    offensive_index = [(sim, team) for sim, team in enumerate(offensive_teams)]
    turnover_probs = on_floor_df.loc[
        on_floor_df.index.droplevel("PlayerID").isin(offensive_index), turnover_fields
    ]
    # let's try truly modeling as an exponential distribution.
    # first calculate rate parameter (1 / theta or beta), aka
    # (1 / average amount of seconds between two events)
    # factor of 2 * tempo_factor because you can only steal
    # when you're playing on defense!
    turnover_probs["turnover_chance_exp_theta"] = 1 / (
        (2 * tempo_factor)
        * turnover_probs["Turnovers"]
        / turnover_probs["Minutes"]
        / 60
    )
    # now use rate parameter to calculate CDF per player, per possession.
    # x = the percentage of the team's gametime that has elapsed
    # numpy array is expanded 5x so each player of the 5 on this side can get their time
    turnover_probs["no_turnover_chance"] = np.e ** (
        -1
        * np.repeat(possession_length, 5)
        / turnover_probs["turnover_chance_exp_theta"]
    )
    # we could also randomly sample an exponential for each player using numpy
    # and see if it's less than the number of possession_seconds. might be
    # interesting to see how this changes the simulation.
    turnover_probs["turnover_chance"] = 1 - turnover_probs["no_turnover_chance"]

    return turnover_probs


def rebound_distribution(offensive_teams, defensive_teams, on_floor_df):
    off_reb_fields = ["OffensiveRebounds", "sim_offensive_rebounds"]
    def_reb_fields = ["DefensiveRebounds", "sim_defensive_rebounds"]

    # dropping the PlayerID level allows us to slice out only the teams on each side in each sim.
    offensive_index = [(sim, team) for sim, team in enumerate(offensive_teams)]
    defensive_index = [(sim, team) for sim, team in enumerate(defensive_teams)]
    offensive_rebound_probs = on_floor_df.loc[
        on_floor_df.index.droplevel("PlayerID").isin(offensive_index), off_reb_fields
    ]
    defensive_rebound_probs = on_floor_df.loc[
        on_floor_df.index.droplevel("PlayerID").isin(defensive_index), def_reb_fields
    ]

    team_off_reb_total = offensive_rebound_probs.groupby(level=0).sum()
    team_def_reb_total = defensive_rebound_probs.groupby(level=0).sum()
    rebound_denominator = (
        team_off_reb_total["OffensiveRebounds"][0]
        + team_def_reb_total["DefensiveRebounds"][0]
    )

    off_reb_share = offensive_rebound_probs.div(team_off_reb_total, level="Team")
    offensive_rebound_probs["off_reb_pdf"] = off_reb_share["OffensiveRebounds"]
    def_reb_share = defensive_rebound_probs.div(team_def_reb_total, level="Team")
    defensive_rebound_probs["def_reb_pdf"] = def_reb_share["DefensiveRebounds"]

    off_reb_chance = team_off_reb_total["OffensiveRebounds"][0] / rebound_denominator

    return (
        offensive_rebound_probs,
        defensive_rebound_probs,
        off_reb_chance,
    )


@ab_api.get(
    "/FantasyDataRefresh/PlayerGameDay/{game_year}/{game_month}/{game_day}",
    dependencies=[Depends(oauth2_scheme)],
)
async def refresh_fd_player_games(
    game_year: int,
    game_month: int,
    game_day: int,
    client: AsyncIOMotorClient = Depends(get_odm),
):
    try:
        game_date = date(game_year, game_month, game_day)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"API error: {e}")

    requested_date = game_date.strftime("%Y-%b-%d")

    r = requests.get(
        f"https://api.sportsdata.io/api/cbb/fantasy/json/PlayerGameStatsByDate/{requested_date}"
        + "?key="
        + FANTASY_DATA_KEY_FREE
    )

    engine = AIOEngine(motor_client=client, database="autobracket")

    return {"message": "Mongo refresh complete!"}


@ab_api.get(
    "/FantasyDataRefresh/PlayerSeason/{season}", dependencies=[Depends(oauth2_scheme)]
)
async def refresh_fd_player_season(
    season: FantasyDataSeason,
    client: AsyncIOMotorClient = Depends(get_odm),
):
    r = requests.get(
        f"https://api.sportsdata.io/api/cbb/fantasy/json/PlayerSeasonStats/{season}"
        + "?key="
        + FANTASY_DATA_KEY_FREE
    )

    engine = AIOEngine(motor_client=client, database="autobracket")

    # data manipulation is easier in Pandas!
    player_season_df = pandas.DataFrame(r.json())

    # season should be string (ex: 2020POST). then convert other columns.
    # if we do the opposite order the Season column will throw an error.
    # can't go directly from int to str - you'll get an error! map first
    player_season_df["Season"] = player_season_df["Season"].map(str).astype("string")
    player_season_df = player_season_df.convert_dtypes()

    # position is None for about 3200 players...fill with "Not Found"
    player_season_df["Position"] = player_season_df["Position"].fillna("Not Found")

    # renaming a few fields - we're going to overwrite them with more exact percentages next step
    player_season_df.rename(
        columns={
            "TwoPointersPercentage": "two_chance",
            "ThreePointersPercentage": "three_chance",
            "FreeThrowsPercentage": "ft_chance",
        },
        inplace=True,
    )

    # (re-)calculated fields for use in analysis. need to cast to float for division
    # to work properly, then fillna with zero
    player_season_df["two_attempt_chance"] = (
        pandas.to_numeric(player_season_df["TwoPointersAttempted"], downcast="float")
        / pandas.to_numeric(player_season_df["FieldGoalsAttempted"], downcast="float")
    ).fillna(0)
    player_season_df["two_chance"] = (
        pandas.to_numeric(player_season_df["TwoPointersMade"], downcast="float")
        / pandas.to_numeric(player_season_df["TwoPointersAttempted"], downcast="float")
    ).fillna(0)
    player_season_df["three_chance"] = (
        pandas.to_numeric(player_season_df["ThreePointersMade"], downcast="float")
        / pandas.to_numeric(
            player_season_df["ThreePointersAttempted"], downcast="float"
        )
    ).fillna(0)
    player_season_df["ft_chance"] = (
        pandas.to_numeric(player_season_df["FreeThrowsMade"], downcast="float")
        / pandas.to_numeric(player_season_df["FreeThrowsAttempted"], downcast="float")
    ).fillna(0)

    # back to json for writing to DB
    p = orjson.loads(player_season_df.to_json(orient="records"))
    await engine.save_all([PlayerSeason(**doc) for doc in p])

    return {"message": "Mongo refresh complete!"}


@ab_api.get(
    "/FantasyDataRefresh/PlayerSeasonTeam/{season}/{team}",
    dependencies=[Depends(oauth2_scheme)],
)
async def refresh_fd_player_season_team(
    season: FantasyDataSeason,
    team: str,
):
    r = requests.get(
        f"https://api.sportsdata.io/api/cbb/fantasy/json/PlayerSeasonStatsByTeam/{season}/{team}"
        + "?key="
        + FANTASY_DATA_KEY_FREE
    )

    return {"message": "Mongo refresh complete!"}
