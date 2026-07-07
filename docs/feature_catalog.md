# Feature Catalog

## Rules

Every feature must be available after draft and before game start.

For historical features:

- sort matches by match date and a stable match id tie-breaker
- compute aggregates from prior matches only
- never include the target match in its own history
- never use matches later than the target match

For missing history:

- count-like features default to `0`
- rate-like features default to a documented prior, usually `0.5` or the training-set global rate

## Target

| Feature | Type | Description | Leakage Risk |
| --- | --- | --- | --- |
| `blue_win` | label | `1` if blue side wins, else `0` | Label only, never used as input |

## Match Context

| Feature | Type | Description | V1 |
| --- | --- | --- | --- |
| `region` | categorical | League or event region such as LCK, LPL, LEC, LCS, INT. International aliases are normalized to INT. | yes |
| `year` | categorical/numeric | Tournament year | yes |
| `split` | categorical | Canonical split such as Winter, Spring, Summer | yes |
| `stage` | categorical | Source stage label such as WEEK1 or playoffs | yes |
| `is_international` | boolean | `1` when region is INT | yes |
| `is_playoffs` | boolean | Derived from stage/tournament text | optional |
| `is_finals` | boolean | Derived from stage/tournament text | optional |

## Draft Control

| Feature | Type | Description | V1 |
| --- | --- | --- | --- |
| `first_pick_side` | categorical | Side with first pick | yes |
| `blue_has_first_pick` | boolean | `1` if blue had first pick | yes |
| `red_has_first_pick` | boolean | `1` if red had first pick | optional, redundant |

## Team Identity

| Feature | Type | Description | V1 |
| --- | --- | --- | --- |
| `blue_team_name` | categorical | Canonical blue team code | yes |
| `red_team_name` | categorical | Canonical red team code | yes |

## Player Identity

Use raw player names as categorical inputs in v1.

| Feature | Type | Description | V1 |
| --- | --- | --- | --- |
| `blue_top_player` | categorical | Blue top player | yes |
| `blue_jungle_player` | categorical | Blue jungle player | yes |
| `blue_mid_player` | categorical | Blue mid player | yes |
| `blue_bot_player` | categorical | Blue bot player | yes |
| `blue_support_player` | categorical | Blue support player | yes |
| `red_top_player` | categorical | Red top player | yes |
| `red_jungle_player` | categorical | Red jungle player | yes |
| `red_mid_player` | categorical | Red mid player | yes |
| `red_bot_player` | categorical | Red bot player | yes |
| `red_support_player` | categorical | Red support player | yes |

Do not add roster movement modeling in v1. The categorical encoder should handle unseen players.

## Champion Picks

| Feature | Type | Description | V1 |
| --- | --- | --- | --- |
| `blue_top_champion` | categorical | Blue top champion | yes |
| `blue_jungle_champion` | categorical | Blue jungle champion | yes |
| `blue_mid_champion` | categorical | Blue mid champion | yes |
| `blue_bot_champion` | categorical | Blue bot champion | yes |
| `blue_support_champion` | categorical | Blue support champion | yes |
| `red_top_champion` | categorical | Red top champion | yes |
| `red_jungle_champion` | categorical | Red jungle champion | yes |
| `red_mid_champion` | categorical | Red mid champion | yes |
| `red_bot_champion` | categorical | Red bot champion | yes |
| `red_support_champion` | categorical | Red support champion | yes |

## Bans

| Feature | Type | Description | V1 |
| --- | --- | --- | --- |
| `blue_bans` | categorical/list | Blue-side banned champions | optional |
| `red_bans` | categorical/list | Red-side banned champions | optional |

V1 can skip bans if encoding them slows the first model. Add them once the pick-based model path works.

## Team Split Form

Compute these before each match within the same region/year/split.

| Feature | Type | Description | V1 |
| --- | --- | --- | --- |
| `blue_team_split_games_before` | numeric | Blue team's prior games in split | yes |
| `blue_team_split_win_rate_before` | numeric | Blue team's prior split win rate | yes |
| `red_team_split_games_before` | numeric | Red team's prior games in split | yes |
| `red_team_split_win_rate_before` | numeric | Red team's prior split win rate | yes |
| `team_split_win_rate_diff` | numeric | Blue win rate minus red win rate | yes |

Leakage rule: update team records only after producing the feature row for the current match.

## Champion Role History

Compute these before each match for the selected champion in the selected role within the same region/year/split.

| Feature Pattern | Type | Description | V1 |
| --- | --- | --- | --- |
| `{side}_{role}_champion_role_games_before` | numeric | Prior games for champion in role within region/year/split | yes |
| `{side}_{role}_champion_role_wins_before` | numeric | Prior wins for champion in role | optional |
| `{side}_{role}_champion_role_win_rate_before` | numeric | Prior win rate for champion in role within region/year/split | yes |
| `{side}_{role}_champion_role_pick_rate_before` | numeric | Prior pick rate for champion in role within split/region | optional |

Example:

- `blue_mid_champion_role_games_before`
- `blue_mid_champion_role_win_rate_before`

## Player-Champion Comfort

These are required in v1.

For each side and role, compute the selected player's history on the selected champion over the previous 2 years.

| Feature Pattern | Type | Description | V1 |
| --- | --- | --- | --- |
| `{side}_{role}_player_champion_games_last_2y` | numeric | Player's prior games on selected champion in the previous 2 years | yes |
| `{side}_{role}_player_champion_win_rate_last_2y` | numeric | Player's prior win rate on selected champion in the previous 2 years | yes |

Examples:

- `blue_mid_player_champion_games_last_2y`
- `blue_mid_player_champion_win_rate_last_2y`
- `red_jungle_player_champion_games_last_2y`
- `red_jungle_player_champion_win_rate_last_2y`

Leakage rule: the current match must not count toward either value.

Fallback:

- no prior games: games = `0`
- no prior wins/losses: win rate = configured prior, initially `0.5`

## Deferred Features

Do not build these until v1 works:

- player pick share
- comfort buckets
- champion composition tags
- engage, poke, dive, scaling, or damage-profile features
- patch-specific champion strength
- player embeddings
- team roster continuity
- automated explainability features

They may improve the model, but they are not needed to prove the pipeline.
