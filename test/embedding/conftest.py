# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/embedding/conftest.py
'''
Fixtures/setup/teardown for embedding tests

General note: After setup as described in the README.md for this directory, run the tests with:

pytest test

or, for just embeddings tests:

pytest test/embedding/
'''

import sys
import os
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, DEFAULT  # noqa: F401

import numpy as np
from ogbujipt.embedding.pgvector import MessageDB, DataDB


@pytest.fixture
def CORRECT_STRING():
    return 'And the secret thing in its heaving\nThreatens with iron mask\nThe last lighted torch of the century…'

# Putting a few more fixtures for test strings for mocking purposes
'''
# Recipe for printing out the embedding 8 numbers across
lines = []
for k, g in groupby(enumerate(emb), lambda x: x[0] // 8):
    lines.append(', '.join([repr(i[1]) for i in g]))
# print(',\n'.join(lines))
# Doesn't work because of numerical imprecision
# check = eval('[' + ',\n'.join(lines) + ']') == emb
'''
# Embedding of 'Hi there!'
@pytest.fixture
def HITHERE_all_MiniLM_L12_v2():
    return np.array(
[-0.06131593, 0.09109506, -0.014503071, 0.037823223, 0.030189564, -0.06783975, 0.09422422, 0.04006801,
-0.0880375, 0.08886486, -0.023855003, -0.017653001, -0.019381238, -0.04008238, -0.0057603996, -0.030874358,
0.068908915, -0.004609795, -0.04091627, -0.043000214, 3.0771265e-05, 0.0021153986, -0.090434276, -0.0061122826,
-0.0045277686, -0.084255695, -0.014587876, -0.03407146, 0.026380118, -0.07527129, 0.038600676, -0.008446554,
0.11756941, -0.011500297, -0.004529579, 0.034317482, 0.029529935, -0.022262357, 0.082906105, 0.01985052,
-0.024071347, 0.032835837, -0.019270986, -0.028544752, 0.039697994, -0.07075252, -0.009630896, -0.022671351,
-0.025061082, 0.022867734, 0.0041197655, 0.048563913, -0.08896639, 0.041883204, -0.0024471097, -0.022467814,
-0.0066597513, -0.050062347, -0.020810338, 0.01503336, -0.062190346, -0.007934369, -0.007602565, -0.04890965,
-0.028964916, 0.01587389, 0.0061304593, 0.011793535, 0.080727786, -0.11510921, -0.021443322, 0.014525274,
0.0131347, -0.050659724, 0.04632249, 0.022516359, 0.002481182, 0.003983038, 0.06503839, 0.006376998,
0.057025786, -0.07867812, -0.0015248525, 0.010740209, 0.024031896, 0.06371934, 0.008501752, 0.028156277,
0.025742045, -0.006634087, -0.06038313, 0.05288649, 0.022682209, -0.03241306, -0.055169605, -0.011404284,
0.02028607, 0.012700554, -0.042174477, 0.1687151, -0.0395818, 0.041207932, -0.014557371, 0.08225274,
-0.1362601, 0.08978466, -0.05852121, -0.03893623, 0.05446341, -0.03584422, 0.028031515, -0.017192937,
-0.115677305, -0.0143916095, 0.01518451, 0.07031115, 0.029574988, 0.07056617, -0.019578027, -0.05384695,
-0.07022203, 0.0008725191, 0.046562, -0.087284446, 0.06583815, -0.06973943, 0.012903563, 0.026464567,
-0.007460593, -0.03897838, 0.06069256, 0.04464523, -0.025860358, 0.022152388, -0.059649926, -0.042634524,
-0.009668905, -0.042315036, 0.035126623, -0.06046377, -0.022582775, 0.03808319, -0.05037486, -0.006364651,
0.03011575, -0.15298544, 0.05122064, 0.013105435, 0.062012393, 0.054373667, 0.03023016, 0.04183447,
0.021090044, 0.05488146, 0.0234279, -0.026130427, 0.032523658, 0.044661526, 0.003953107, 0.020838184,
-0.0066597504, -0.017314808, -0.08413289, -0.0216152, 0.03886422, -0.02728897, 0.037595816, 0.03651264,
0.035920188, -0.041534312, 0.03210632, 0.04540799, 0.0027102598, -0.043059606, 0.1006973, 0.016418878,
-0.019285522, 0.03344562, -4.8311536e-05, 0.04767274, -0.042215683, 0.08040912, -0.08199628, -0.006984787,
-0.011871969, -0.04519371, 0.006704613, 0.081617825, -0.014620735, 0.07301282, 0.015881404, -0.023621494,
-0.06694493, -0.041859355, -0.027709452, -0.022445913, -0.00018571844, 0.06775287, 0.09135706, -0.034682617,
-0.08574087, 0.02100846, 0.019011416, 0.05252803, 0.056879684, -0.06569118, -0.04257552, 0.0148884915,
-0.1082349, -0.034131706, -0.0036737903, 0.030427922, 0.018549278, -0.002457179, -0.023634495, -0.06645388,
0.038994897, 0.05739158, -0.09076464, -0.030154258, 0.059230182, 0.021086855, -0.10302539, 1.4361168e-33,
0.0022506495, 0.029155267, -0.011308517, -0.0564986, 0.023305397, 0.027631093, -0.05894308, 0.07272398,
-0.0034047149, 0.0029925616, -0.03273505, -0.011123168, 0.019808466, -0.053017292, -0.010162193, 0.046002056,
0.013122092, -0.0029093053, 0.02964658, 0.007230795, -0.032439813, 0.024106877, -0.0032050312, -0.08868969,
0.018305168, -0.027682163, 0.08238465, 0.03400411, -0.16178623, 0.009540504, -0.01806741, 0.04289511,
-0.0018772804, 0.012215782, 0.060714856, 0.08925001, -0.013223062, -0.11776229, -0.105936855, -0.04966384,
-0.080789454, -0.07687011, -0.08648494, 0.018879725, -0.0100128455, -0.027153097, -0.06250377, -0.014781195,
-0.010176314, -0.008085249, -0.13907915, -0.013339057, 0.071909115, -0.08558494, -0.055899236, 0.00018726928,
0.011606009, 0.030509377, 0.11078848, -0.0055666906, -0.037034787, 0.08291926, 0.09356686, 0.0624804,
0.049453255, 0.010197718, 0.058185525, -0.035388555, 0.089378506, 0.014276284, -0.0051372945, -0.023549069,
0.07148825, -0.008391381, -0.025205271, 0.057586137, -0.0147248795, -0.050060436, -0.025024073, 0.07353391,
-0.019859206, 0.08820009, -0.027162977, -0.04908677, 0.042011738, 0.04837553, 0.10521715, 0.087377235,
-0.0801893, 0.011782323, 0.092166364, 0.034470957, 0.030874413, -0.043447766, 0.046303295, -3.2446916e-34,
-0.027896123, 0.029717434, 0.08947059, -0.010339605, 0.07417876, 0.039140556, -0.04288442, 0.029881801,
-0.12673245, -0.03935677, 0.041521396, 0.028670557, 0.0031351803, -0.018653063, 0.046313535, -0.026449086,
-0.019898465, 0.053464845, -0.0029477212, -0.060369834, 0.0062588095, 0.015188291, 0.08092363, 0.07501285,
0.04322814, 0.042307224, 0.029623767, -0.06942712, -0.061772335, -0.005188368, -0.024355723, 0.07200778,
-0.023118502, -0.0009832964, -0.074652284, 0.00043690225, -0.0075028646, 0.01459858, -0.0053168857, -0.0974746,
-0.071223356, 0.0019683593, 0.032683503, -0.08899012, 0.10160039, 0.04948275, 0.048017487, -0.046223965,
0.032460734, -0.043729845, 0.030224336, -0.019220904, 0.08223829, 0.03851222, -0.016376046, 0.041965306,
0.0445879, -0.03780432, -0.024826797, 0.014669102, 0.057102628, -0.031820614, 0.0027352672, 0.052658144])

# XXX: This stanza to go away once mocking is complete - Kai
HOST = os.environ.get('PG_HOST', 'localhost')
DB_NAME = os.environ.get('PG_DATABASE', 'mock_db')
USER = os.environ.get('PG_USER', 'mock_user')
PASSWORD = os.environ.get('PG_PASSWORD', 'mock_password')
PORT = os.environ.get('PG_PORT', 5432)


# XXX: Move to a fixture?
# Definitely don't want to even import SentenceTransformer class due to massive side-effects
class SentenceTransformer(object):
    def __init__(self, model_name_or_path):
        self.encode = MagicMock()


DB_CLASS = {
    'test/embedding/test_pgvector_message.py': MessageDB,
    'test/embedding/test_pgvector_data.py': DataDB,
}

# print(HOST, DB_NAME, USER, PASSWORD, PORT)

@pytest_asyncio.fixture  # Notice the async aware fixture declaration
async def DB(request):
    testname = request.node.name
    testfile = request.node.location[0]
    table_name = testname.lower()
    print(f'DB setup for test: {testname}. Table name {table_name}', file=sys.stderr)
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])
    try:
        vDB = await DB_CLASS[testfile].from_conn_params(
            embedding_model=dummy_model,
            table_name=table_name,
            db_name=DB_NAME,
            host=HOST,
            port=int(PORT),
            user=USER,
            password=PASSWORD)
    except ConnectionRefusedError:
        pytest.skip("No Postgres instance made available for test. Skipping.", allow_module_level=True)
    # Actually we want to propagate the error condition, in this case
    # if vDB is None:
    #     pytest.skip("Unable to create a valid DB instance. Skipping.", allow_module_level=True)

    # Create table
    await vDB.drop_table()
    assert not await vDB.table_exists(), Exception("Table exists before creation")
    await vDB.create_table()
    assert await vDB.table_exists(), Exception("Table does not exist after creation")
    # The test will take control upon the yield
    yield vDB
    # Teardown: Drop table
    await vDB.drop_table()


# FIXME: Lots of DRY violations
@pytest_asyncio.fixture  # Notice the async aware fixture declaration
async def DB_WINDOWED2(request):
    testname = request.node.name
    table_name = testname.lower()
    print(f'DB setup for test: {testname}. Table name {table_name}', file=sys.stderr)
    dummy_model = SentenceTransformer('mock_transformer')
    dummy_model.encode.return_value = np.array([1, 2, 3])
    try:
        vDB = await MessageDB.from_conn_params(
            embedding_model=dummy_model,
            table_name=table_name,
            db_name=DB_NAME,
            host=HOST,
            port=int(PORT),
            user=USER,
            password=PASSWORD,
            window=2)
    except ConnectionRefusedError:
        pytest.skip("No Postgres instance made available for test. Skipping.", allow_module_level=True)
    # Actually we want to propagate the error condition, in this case
    # if vDB is None:
    #     pytest.skip("Unable to create a valid DB instance. Skipping.", allow_module_level=True)

    # Create table
    await vDB.drop_table()
    assert not await vDB.table_exists(), Exception("Table exists before creation")
    await vDB.create_table()
    assert await vDB.table_exists(), Exception("Table does not exist after creation")
    # The test will take control upon the yield
    yield vDB
    # Teardown: Drop table
    await vDB.drop_table()
