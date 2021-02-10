# import native Python packages
from datetime import date
from enum import Enum
import multiprocessing
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

# import custom local stuff
from instance.config import FANTASY_DATA_KEY_FREE
from src.db.atlas import get_odm
from src.api.users import oauth2_scheme


ab_api = APIRouter(
    prefix="/autobracket",
    tags=["autobracket"],
    dependencies=[Depends(oauth2_scheme)],
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
    TwoPointersPercentage: float
    ThreePointersMade: int
    ThreePointersAttempted: int
    ThreePointersPercentage: float
    FreeThrowsMade: int
    FreeThrowsAttempted: int
    FreeThrowsPercentage: float
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


@ab_api.get("/stats/{season}/all")
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


@ab_api.get("/stats/{season}/{team}")
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
    ).set_index(
        ["Team", "PlayerID"]
    )

    # calculate potential columns for clustering, drop others
    player_df["two_attempt_rate"] = player_df["TwoPointersAttempted"] / player_df["FieldGoalsAttempted"]
    player_df["two_success_rate"] = player_df["TwoPointersMade"] / player_df["TwoPointersAttempted"]
    player_df["three_success_rate"] = player_df["ThreePointersMade"] / player_df["ThreePointersAttempted"]
    player_df["ft_success_rate"] = player_df["FreeThrowsMade"] / player_df["FreeThrowsAttempted"]

    player_df["points_per_second"] = player_df["Points"] / player_df["Minutes"] / 60
    player_df["shots_per_second"] = player_df["FieldGoalsAttempted"] / player_df["Minutes"] / 60
    player_df["rebounds_per_second"] = player_df["Rebounds"] / player_df["Minutes"] / 60
    player_df["assists_per_second"] = player_df["Assists"] / player_df["Minutes"] / 60
    player_df["steals_per_second"] = player_df["Steals"] / player_df["Minutes"] / 60
    player_df["blocks_per_second"] = player_df["BlockedShots"] / player_df["Minutes"] / 60
    player_df["turnovers_per_second"] = player_df["Turnovers"] / player_df["Minutes"] / 60
    player_df["fouls_per_second"] = player_df["PersonalFouls"] / player_df["Minutes"] / 60

    # drop anyone that didn't play a minute or has null values (could be took no shots etc.)
    player_df = player_df.loc[player_df["Minutes"] > 0].dropna()

    # work in progress, but these will be the columns to start with
    player_df = player_df[[
        "two_attempt_rate",
        "two_success_rate",
        "three_success_rate",
        "ft_success_rate",
        # "points_per_second",
        # "shots_per_second",
        # "rebounds_per_second",
        # "assists_per_second",
        # "steals_per_second",
        # "blocks_per_second",
        # "turnovers_per_second",
        # "fouls_per_second",
    ]]

    # columns for the scatter plot (do this before adding labels to the data)
    scatter_cols = player_df.columns.tolist()

    # K-Means time! 10 pretty much looks like where the elbow tapers off,
    # when looking at the four "rate" variables.
    model = KMeans(n_clusters=10)
    model.fit(player_df)
    player_df["player_type"] = model.predict(player_df)

    # for now just display 1/10th of the data
    player_df = player_df.sample(
        frac=0.1, replace=False,
    )

    # remove outliers (these are probably folks with very few minutes anyway)
    player_df = player_df.loc[(np.abs(stats.zscore(player_df)) < 3).all(axis=1)]

    scatter_data = orjson.loads(player_df.to_json(orient="records"))
    
    return {
        'data': scatter_data,
        'columns': scatter_cols,
        'inertia': model.inertia_
    }


@ab_api.get("/sim/{season}/{team_one}/{team_two}/{sample_size}")
async def full_game_simulation(
    season: FantasyDataSeason,
    team_one: str,
    team_two: str,
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
            & ((PlayerSeason.Team == team_one) | (PlayerSeason.Team == team_two)),
            sort=(PlayerSeason.Team, PlayerSeason.StatID),
        )
    ]

    # create a dataframe representing one simulation
    matchup_df = pandas.DataFrame(
        [player_season.doc() for player_season in matchup_data]
    )

    # create a list of matchup dfs representing multiple simulations
    if sample_size > 1:
        cores_to_use = multiprocessing.cpu_count()
        simulations = [matchup_df.copy() for x in range(sample_size)]

        with multiprocessing.Pool(processes=cores_to_use) as p:
            results = p.map(run_simulation, simulations)
            # clean up
            p.close()
            p.join()
    else:
        # just do one run
        results = run_simulation(matchup_df)

    end_time = perf_counter()

    return {
        "time": (end_time - start_time),
        "simulations": sample_size,
        "results": results,
    }


def run_simulation(matchup_df):
    # assign columns for shot distributions and simulated game stats
    matchup_df = matchup_df.assign(
        two_attempt_chance=lambda x: x.TwoPointersAttempted / x.FieldGoalsAttempted,
        two_chance=lambda x: x.TwoPointersMade / x.TwoPointersAttempted,
        three_chance=lambda x: x.ThreePointersMade / x.ThreePointersAttempted,
        ft_chance=lambda x: x.FreeThrowsMade / x.FreeThrowsAttempted,
        sim_seconds=0,
        sim_two_pointers_made=0,
        sim_two_pointers_attempted=0,
        sim_three_pointers_made=0,
        sim_three_pointers_attempted=0,
        sim_free_throws_made=0,
        sim_free_throws_attempted=0,
        sim_offensive_rebounds=0,
        sim_defensive_rebounds=0,
        sim_assists=0,
        sim_steals=0,
        sim_blocks=0,
        sim_turnovers=0,
        sim_fouls=0,
    )

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

    # new numpy random number generator
    rng = np.random.default_rng()

    # determine first possession (simple 50/50 for now)
    matchup_list = team_minute_totals.index.to_list()
    possession_flag = rng.integers(2, size=1)[0]

    # game clock start, shot clock reset flag, initialize possession length,
    # possession counter for each team
    time_remaining = 60 * 40
    shot_clock_reset = True
    possession_length = 0
    total_possessions = 0
    tempo_factor = 1 / 1

    # set index that will be the basis for updating box score.
    matchup_df.set_index(["Team", "PlayerID"], inplace=True)

    while time_remaining >= 0:
        # who has the ball?
        offensive_team = matchup_list[possession_flag]
        defensive_team = matchup_list[(1 - possession_flag)]

        if time_remaining == 0:
            # game might be over. calculate score and see if we need OT
            matchup_df = matchup_df.assign(
                sim_points=lambda x: (
                    (x.sim_free_throws_made)
                    + (x.sim_two_pointers_made * 2)
                    + (x.sim_three_pointers_made * 3)
                ),
            )
            # aggregate team box score
            team_score_df = matchup_df.groupby(level=0).agg({"sim_points": "sum"})
            if (
                team_score_df.loc[offensive_team][0]
                == team_score_df.loc[defensive_team][0]
            ):
                # start a 5 minute overtime!
                time_remaining = 60 * 5
                continue
            else:
                # end loop!
                break

        # possession length simulated here as uniform from 5-30 seconds, but
        # we definitely need to pull in some sort of tempo per team here.
        if shot_clock_reset:
            # possession counter
            total_possessions += 1
            possession_length = min(
                [
                    (30 - 5) * rng.random() + 5,
                    time_remaining,
                ]
            )
        else:
            # shot clock didn't reset, so just use part of the leftover seconds
            # an improvement here would be to lower the likelihood of a successful
            # offensive possession
            possession_length = min(
                [
                    (30 - possession_length) * rng.random(),
                    time_remaining,
                ]
            )
            shot_clock_reset = not shot_clock_reset

        # pick 10 players for the current possession based on average time share
        on_floor_df = matchup_df.groupby(level=0).sample(
            n=5, replace=False, weights=matchup_df.minute_dist.to_list()
        )

        # add the possession length to the time played for each individual on the floor and update
        on_floor_df["sim_seconds"] = on_floor_df["sim_seconds"] + possession_length
        matchup_df.update(on_floor_df)

        # now, based on the 10 players on the floor, calculate probability of each event.
        # first, a steal check happens here. use steals per second over the season.
        # improvement: factor in the opponent's turnover statistics here.
        # steals per second times possession length to get the steal chance for this possession
        # times 2 assumes each team has the ball for about half the game. this effectively
        # converts steals per both teams' possession to steals per defensive possession (since
        # you can't get a steal while you're on offense!)
        # i think this is where we would put a tempo factor...
        steal_probs = steal_distribution(possession_length, defensive_team, on_floor_df, tempo_factor)

        # we also need turnover probabilities here
        turnover_probs = turnover_distribution(
            possession_length, offensive_team, on_floor_df, tempo_factor
        )

        # the steal/turnover check! we're modeling them as independent.
        # (right now it's possible that a turnover in a given possession
        # will always be a steal, if turnover_chance is less than steal_chance.)
        steal_turnover_success = rng.random()
        steal_chance = steal_probs.loc[defensive_team].steal_chance_cdf.max()
        turnover_chance = turnover_probs.loc[offensive_team].turnover_chance_cdf.max()

        # if there's a successful steal, credit the steal and turnover, then flip possession and restart loop!
        if steal_turnover_success < steal_chance:
            # who got the steal?
            steal_player = steal_probs.sample(n=1, weights=steal_probs.steal_chance_cdf)
            steal_player["sim_steals"] = steal_player["sim_steals"] + 1
            matchup_df.update(steal_player)

            # who committed the turnover?
            turnover_player = turnover_probs.sample(
                n=1, weights=turnover_probs.turnover_chance_cdf
            )
            turnover_player["sim_turnovers"] = turnover_player["sim_turnovers"] + 1
            matchup_df.update(turnover_player)

            # change clock, give ball to other team, reset loop
            time_remaining -= possession_length
            possession_flag = 1 - possession_flag
            continue
        # if there's a turnover, credit the turnover, then flip possession and restart loop!
        elif steal_turnover_success < turnover_chance:
            # who committed the turnover?
            turnover_player = turnover_probs.sample(
                n=1, weights=turnover_probs.turnover_chance_cdf
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

        # if a defensive player blocks, 50/50 chance to be a turnover.
        # using blocks per second over the season.
        # we're either crediting miss+block, or miss+block+rebound.
        block_probs = block_distribution(possession_length, defensive_team, on_floor_df, tempo_factor)

        # block check!
        block_success = rng.random()
        block_chance = block_probs.block_chance_cdf.max()

        # the shot type check!
        two_or_three = rng.random()

        if block_success < block_chance:
            # who got the block?
            blocking_player = block_probs.sample(
                n=1, weights=block_probs.block_chance_cdf
            )
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

        # if we've made it this far, the shot was not blocked.
        # but did it go in? and was there a foul?
        foul_probs = foul_distribution(possession_length, defensive_team, on_floor_df, tempo_factor)

        # defensive foul check! (potential improvement, offensive fouls and
        # non-shooting fouls)
        foul_chance = foul_probs.loc[defensive_team].foul_chance_cdf.max()
        foul_occurred = rng.random()

        if two_or_three < shooting_player.two_attempt_chance.values[0]:
            # check two point probability
            two_success = rng.random()
            if two_success < shooting_player.two_chance.values[0]:
                # credit the shooter with a 2pt attempt and make
                shooting_player["sim_two_pointers_attempted"] = (
                    shooting_player["sim_two_pointers_attempted"] + 1
                )
                shooting_player["sim_two_pointers_made"] = (
                    shooting_player["sim_two_pointers_made"] + 1
                )
                if foul_occurred < foul_chance:
                    # and-1! first, credit the foul
                    fouling_player = foul_probs.sample(
                        n=1, weights=foul_probs.foul_chance_cdf
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
                assist_probs = assist_distribution(
                    possession_length, offensive_team, on_floor_df, tempo_factor
                )

                # assist check!
                assist_success = rng.random()
                assist_chance = assist_probs.assist_chance_cdf.max()

                if assist_success < assist_chance:
                    # who got the assist?
                    assisting_player = assist_probs.sample(
                        n=1, weights=assist_probs.assist_chance_cdf
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
                if foul_occurred < foul_chance:
                    # if fouled, two FTs, and shooter is NOT credited with a missed shot
                    fouling_player = foul_probs.sample(
                        n=1, weights=foul_probs.foul_chance_cdf
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
            if three_success < shooting_player.three_chance.values[0]:
                # credit the shooter with a 2pt attempt and make
                shooting_player["sim_three_pointers_attempted"] = (
                    shooting_player["sim_three_pointers_attempted"] + 1
                )
                shooting_player["sim_three_pointers_made"] = (
                    shooting_player["sim_three_pointers_made"] + 1
                )
                if foul_occurred < foul_chance:
                    # and-1! first, credit the foul
                    fouling_player = foul_probs.sample(
                        n=1, weights=foul_probs.foul_chance_cdf
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
                assist_probs = assist_distribution(
                    possession_length, offensive_team, on_floor_df, tempo_factor
                )

                # assist check!
                assist_success = rng.random()
                assist_chance = assist_probs.assist_chance_cdf.max()

                if assist_success < assist_chance:
                    # who got the assist?
                    assisting_player = assist_probs.sample(
                        n=1, weights=assist_probs.assist_chance_cdf
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
                if foul_occurred < foul_chance:
                    # if fouled, three FTs, and shooter is NOT credited with a missed shot
                    fouling_player = foul_probs.sample(
                        n=1, weights=foul_probs.foul_chance_cdf
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

    # calculate totals
    box_score_df = box_score_df.assign(sim_minutes=lambda x: x.sim_seconds / 60)

    # aggregate team box score
    team_box_score_df = box_score_df.groupby(level=0).sum()

    box_score_json = orjson.loads(box_score_df.to_json(orient="index"))
    team_box_score_json = orjson.loads(team_box_score_df.to_json(orient="index"))

    return [total_possessions, team_box_score_json, box_score_json]


def assist_distribution(possession_length, offensive_team, on_floor_df, tempo_factor):
    assist_fields = [
        "Assists",
        "Minutes",
        "sim_assists",
    ]
    assist_probs = on_floor_df.loc[[offensive_team], assist_fields]
    # alternative...let's try truly modeling as an exponential distribution.
    # first calculate rate parameter (per game estimate)
    assist_probs["assist_chance_exp_lambda"] = (
        assist_probs["Assists"] / assist_probs["Minutes"] * 20 * tempo_factor
    )
    # now use rate parameter to calculate CDF per player, per possession.
    # x = the percentage of the team's gametime that has elapsed
    assist_probs["assist_chance_cdf"] = 1 - np.e ** (
        -1
        * assist_probs["assist_chance_exp_lambda"]
        * possession_length
        / (20 * tempo_factor)
    )
    return assist_probs


def foul_distribution(possession_length, defensive_team, on_floor_df, tempo_factor):
    foul_fields = [
        "PersonalFouls",
        "Minutes",
        "sim_fouls",
    ]
    foul_probs = on_floor_df.loc[[defensive_team], foul_fields]

    # the x2 factor is still here for fouls, even though you can foul on offense or defense.
    # but for now this model is assuming you can only foul on defense. we would scrap
    # this factor once we're modeling offensive fouls independently
    # foul_probs["foul_chance_pdf"] = (
    #     foul_probs["PersonalFouls"] / foul_probs["Minutes"] / 60 * possession_length * 2
    # )
    # foul_probs["foul_chance_cdf"] = foul_probs.groupby(level=0).cumsum()[
    #     "foul_chance_pdf"
    # ]
    # alternative...let's try truly modeling as an exponential distribution.
    # first calculate rate parameter (fouls per game estimate)
    foul_probs["foul_chance_exp_lambda"] = (
        foul_probs["PersonalFouls"] / foul_probs["Minutes"] * 20 * tempo_factor
    )
    # now use rate parameter to calculate CDF per player, per possession.
    # x = the percentage of the team's gametime that has elapsed
    foul_probs["foul_chance_cdf"] = 1 - np.e ** (
        -1
        * foul_probs["foul_chance_exp_lambda"]
        * possession_length
        / (20 * tempo_factor)
    )
    return foul_probs


def block_distribution(possession_length, defensive_team, on_floor_df, tempo_factor):
    block_fields = ["BlockedShots", "Minutes", "sim_blocks"]
    block_probs = on_floor_df.loc[[defensive_team], block_fields]
    # alternative...let's try truly modeling as an exponential distribution.
    # first calculate rate parameter (per game estimate)
    block_probs["block_chance_exp_lambda"] = (
        block_probs["BlockedShots"] / block_probs["Minutes"] * 20 * tempo_factor
    )
    # now use rate parameter to calculate CDF per player, per possession.
    # x = the percentage of the team's gametime that has elapsed
    block_probs["block_chance_cdf"] = 1 - np.e ** (
        -1
        * block_probs["block_chance_exp_lambda"]
        * possession_length
        / (20 * tempo_factor)
    )
    return block_probs


def identify_shooter(offensive_team, on_floor_df):
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


def steal_distribution(possession_length, defensive_team, on_floor_df, tempo_factor):
    steal_fields = [
        "Steals",
        "Minutes",
        "sim_steals",
    ]
    steal_probs = on_floor_df.loc[[defensive_team], steal_fields]
    steal_probs["steal_chance_pdf"] = (
        steal_probs["Steals"] / steal_probs["Minutes"] / 60 * possession_length * 2
    )
    steal_probs["steal_chance_cdf"] = steal_probs.groupby(level=0).cumsum()[
        "steal_chance_pdf"
    ]
    # alternative...let's try truly modeling as an exponential distribution.
    # first calculate rate parameter (1 / average amount of seconds between two events)
    steal_probs["steal_chance_exp_lambda"] = (
        steal_probs["Steals"] / steal_probs["Minutes"] / 60
    )
    # now use rate parameter to calculate CDF per player, per possession.
    # x = the percentage of the team's gametime that has elapsed
    steal_probs["steal_chance_cdf_new"] = 1 - np.e ** (
        -1
        * steal_probs["steal_chance_exp_lambda"]
        * possession_length
    )
    return steal_probs


def turnover_distribution(possession_length, offensive_team, on_floor_df, tempo_factor):
    turnover_fields = [
        "Turnovers",
        "Minutes",
        "sim_turnovers",
    ]
    turnover_probs = on_floor_df.loc[[offensive_team], turnover_fields]
    # alternative...let's try truly modeling as an exponential distribution.
    # first calculate rate parameter (per game estimate)
    turnover_probs["turnover_chance_exp_lambda"] = (
        turnover_probs["Turnovers"] / turnover_probs["Minutes"] * 20 * tempo_factor
    )
    # now use rate parameter to calculate CDF per player, per possession.
    # x = the percentage of the team's gametime that has elapsed
    turnover_probs["turnover_chance_cdf"] = 1 - np.e ** (
        -1
        * turnover_probs["turnover_chance_exp_lambda"]
        * possession_length
        / (20 * tempo_factor)
    )
    return turnover_probs


def rebound_distribution(offensive_team, defensive_team, on_floor_df):
    off_reb_fields = ["OffensiveRebounds", "sim_offensive_rebounds"]
    def_reb_fields = ["DefensiveRebounds", "sim_defensive_rebounds"]

    offensive_rebound_probs = on_floor_df.loc[[offensive_team], off_reb_fields]
    defensive_rebound_probs = on_floor_df.loc[[defensive_team], def_reb_fields]

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


@ab_api.get("/FantasyDataRefresh/PlayerGameDay/{game_year}/{game_month}/{game_day}")
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


@ab_api.get("/FantasyDataRefresh/PlayerSeason/{season}")
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

    # back to json for writing to DB
    p = orjson.loads(player_season_df.to_json(orient="records"))
    await engine.save_all([PlayerSeason(**doc) for doc in p])

    return {"message": "Mongo refresh complete!"}


@ab_api.get("/FantasyDataRefresh/PlayerSeasonTeam/{season}/{team}")
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
