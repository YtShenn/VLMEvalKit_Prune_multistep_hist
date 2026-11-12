import pandas as pd

from vlmeval.inference import build_guiodyssey_trajectory_indices


def test_guiodyssey_trajectory_sharding_keeps_groups_ordered_and_balanced():
    data = pd.DataFrame(
        {
            'image': [
                '/screenshots/traj_b_1.png',
                '/screenshots/traj_a_2.png',
                '/screenshots/traj_b_0.png',
                '/screenshots/traj_a_0.png',
                '/screenshots/traj_c_0.png',
                '/screenshots/traj_a_1.png',
                '/screenshots/traj_c_1.png',
            ]
        }
    )

    shards = [build_guiodyssey_trajectory_indices(data, rank, 3) for rank in range(3)]

    assert sorted(position for shard in shards for position in shard) == list(range(len(data)))
    owners = {
        position: rank
        for rank, shard in enumerate(shards)
        for position in shard
    }
    assert owners[0] == owners[2]  # traj_b
    assert owners[1] == owners[3] == owners[5]  # traj_a
    assert owners[4] == owners[6]  # traj_c

    for shard in shards:
        by_trajectory = {}
        for position in shard:
            stem = data.iloc[position]['image'].rsplit('/', 1)[-1].rsplit('.', 1)[0]
            trajectory, step = stem.rsplit('_', 1)
            by_trajectory.setdefault(trajectory, []).append(int(step))
        assert all(steps == sorted(steps) for steps in by_trajectory.values())

    loads = [len(shard) for shard in shards]
    assert max(loads) - min(loads) <= 1
