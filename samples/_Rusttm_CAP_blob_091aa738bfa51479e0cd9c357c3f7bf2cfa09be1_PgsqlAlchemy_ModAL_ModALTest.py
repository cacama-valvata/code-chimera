from PgsqlAlchemy.ModAL.ModALMainClass import ModALMainClass

class ModALTest(ModALMainClass):
    def __init__(self):
        super

    def insert_new_row():
        new_row = {'meta': {
            'href': 'https://online.moysklad.ru/api/remap/1.2/report/counterparty/fe19e2a6-ac85-11eb-0a80-09cb003e951f',
            'metadataHref': 'https://online.moysklad.ru/api/remap/1.2/report/counterparty/metadata',
            'type': 'counterparty',
            'mediaType': 'application/json',
            'uuidHref': 'https://online.moysklad.ru/app/#company/edit?id=fe19e2a6-ac85-11eb-0a80-09cb003e951f'
        },
            'counterparty': {
                'meta': {
                    'href': 'https://online.moysklad.ru/api/remap/1.2/entity/counterparty/fe19e2a6-ac85-11eb-0a80-09cb003e951f',
                    'metadataHref': 'https://online.moysklad.ru/api/remap/1.2/entity/counterparty/metadata',
                    'type': 'counterparty',
                    'mediaType': 'application/json',
                    'uuidHref': 'https://online.moysklad.ru/app/#company/edit?id=fe19e2a6-ac85-11eb-0a80-09cb003e951f'
                },
                'id': 'fe19e2a6-ac85-11eb-0a80-09cb003e951f',
                'name': 'ООО "АЛТАЙ-КАБЕЛЬ"',
                'externalCode': 'IKIG5lR8jcemsm7wdSpjW1',
                'email': 'krasulin@altayok.ru',
                'phone': '8-3852-226-677',
                'inn': '2222796034',
                'companyType': 'legal'
            },
            'firstDemandDate': '2021-05-04 11:30:00.000',
            'lastDemandDate': '2023-02-03 05:03:00.000',
            'demandsCount': 6,
            'demandsSum': 17044990,
            'averageReceipt': 2840831.6666666665,
            'returnsCount': 0,
            'returnsSum': 0.0,
            'discountsSum': 215710.0,
            'balance': 20.0,
            'profit': 9256967.0,
            'lastEventDate': None,
            'lastEventText': None,
            'updated': '2023-05-03 16:33:20.944'
        }
        new_cust_bal_row = ModALBaseCustBal(**new_row)

        # version1 insert new line in table
        Session = sessionmaker(bind=engine)
        session = Session()
        # check is presence position?
        qry_object = session.query(ModALBaseCustBal).where(
            ModALBaseCustBal.counterparty == new_cust_bal_row.counterparty)
        if qry_object.first() is None:
            session.add(new_cust_bal_row)
        else:
            qry_object.update(new_row)
        session.commit()

        # version2
        ins = insert(ModALBaseCustBal).values(new_row)
        nothing_on_conflict = ins.on_conflict_do_nothing()
        conn = engine.connect()
        conn.execute(nothing_on_conflict)
        # check position
        res = session.query(ModALBaseCustBal).all()
        for r in res:
            print({
                'position_id': r.position_id,
                'counterparty': r.counterparty,
                'balance': r.balance
            })

    def multiply_insertions():
        new_pos1 = {
            'meta': {
                'href': 'https://online.moysklad.ru/api/remap/1.2/report/counterparty/fe19e2a6-ac85-11eb-0a80-09cb003e951f',
                'metadataHref': 'https://online.moysklad.ru/api/remap/1.2/report/counterparty/metadata',
                'type': 'counterparty',
                'mediaType': 'application/json',
                'uuidHref': 'https://online.moysklad.ru/app/#company/edit?id=fe19e2a6-ac85-11eb-0a80-09cb003e951f'
            },
            'counterparty': {
                'meta': {
                    'href': 'https://online.moysklad.ru/api/remap/1.2/entity/counterparty/fe19e2a6-ac85-11eb-0a80-09cb003e951f',
                    'metadataHref': 'https://online.moysklad.ru/api/remap/1.2/entity/counterparty/metadata',
                    'type': 'counterparty',
                    'mediaType': 'application/json',
                    'uuidHref': 'https://online.moysklad.ru/app/#company/edit?id=fe19e2a6-ac85-11eb-0a80-09cb003e951f'
                },
                'id': 'fe19e2a6-ac85-11eb-0a80-09cb003e951f',
                'name': 'ООО "АЛТАЙ-КАБЕЛЬ"',
                'externalCode': 'IKIG5lR8jcemsm7wdSpjW1',
                'email': 'krasulin@altayok.ru',
                'phone': '8-3852-226-677',
                'inn': '2222796034',
                'companyType': 'legal'
            },
            'firstDemandDate': '2021-05-04 11:30:00.000',
            'lastDemandDate': '2023-02-03 05:03:00.000',
            'demandsCount': 6,
            'demandsSum': 17044990,
            'averageReceipt': 2840831.6666666665,
            'returnsCount': 0,
            'returnsSum': 0.0,
            'discountsSum': 215710.0,
            'balance': 15.0,
            'profit': 9256967.0,
            'lastEventDate': None,
            'lastEventText': None,
            'updated': '2023-05-03 16:33:20.944'
        }

        new_pos2 = {
            'meta': {
                'href': 'https://online.moysklad.ru/api/remap/1.2/report/counterparty/02a517d3-a78b-11ed-0a80-10870008fd53',
                'metadataHref': 'https://online.moysklad.ru/api/remap/1.2/report/counterparty/metadata',
                'type': 'counterparty',
                'mediaType': 'application/json',
                'uuidHref': 'https://online.moysklad.ru/app/#company/edit?id=02a517d3-a78b-11ed-0a80-10870008fd53'},
            'counterparty': {'meta': {
                'href': 'https://online.moysklad.ru/api/remap/1.2/entity/counterparty/02a517d3-a78b-11ed-0a80-10870008fd53',
                'metadataHref': 'https://online.moysklad.ru/api/remap/1.2/entity/counterparty/metadata',
                'type': 'counterparty', 'mediaType': 'application/json',
                'uuidHref': 'https://online.moysklad.ru/app/#company/edit?id=02a517d3-a78b-11ed-0a80-10870008fd53'},
                'id': '02a517d3-a78b-11ed-0a80-10870008fd53', 'name': 'ООО "АСК"',
                'externalCode': 'GrQQb4tzggfccJy3866zF2', 'email': 'info@ask66.ru',
                'phone': '+7 (343) 289-21-82', 'inn': '6678000288', 'companyType': 'legal'},
            'firstDemandDate': None,
            'lastDemandDate': None,
            'demandsCount': 0,
            'demandsSum': 0.0,
            'averageReceipt': 0.0,
            'returnsCount': 0,
            'returnsSum': 0.0,
            'discountsSum': 0.0,
            'balance': 15.0,
            'profit': 0.0,
            'lastEventDate': None,
            'lastEventText': None,
            'updated': '2023-02-08 11:31:40.471'}

        # version1 doesnt allow insert equal rows
        # from https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#postgresql-insert-on-conflict
        # from https://stackoverflow.com/questions/73761641/in-sql-alchemy-how-to-use-on-conflict-do-update-when-one-use-a-dictionary-for
        multy_ins = insert(ModALBaseCustBal).values([new_pos1, new_pos2])
        conn = engine.connect()
        conn.execute(multy_ins)
        print(multy_ins.returning(ModALBaseCustBal.balance))

        # version2 also doesnt insert duplicates
        # DBSession = scoped_session(sessionmaker(bind=engine))
        # DBSession.bulk_insert_mappings(ModALBaseCustBal, [new_pos1, new_pos2])
        # DBSession.commit()

        # version3 also doesnt insert duplicates
        # session = sessionmaker(bind=engine)()
        # new_obj1 = ModALBaseCustBal(**new_pos1)
        # res = session.merge(new_obj1)
        # session.commit()
        # print(res)

        # version 4 also
        # ins = insert(ModALBaseCustBal).values([new_pos1, new_pos2])
        # upd_on_conflict = ins.on_conflict_do_update(constraint='unique_key', set_={col: getattr(ins.excluded, col) for col in new_pos1.keys()})
        # ins = ins.on_conflict_do_update(constraint='table_pk', set_={col: getattr(ins.excluded, col) for col in new_pos1})
        # upd_on_conflict = ins.on_conflict_do_update(constraint='customers_bal_model_counterparty_key', set_=dict(**new_pos1))
        # upd_on_conflict = ins.on_conflict_do_update(constraint='customers_bal_model_counterparty_key', set_=dict(balance=ins.excluded.balance))
        # upd_on_conflict = ins.on_conflict_do_update(index_elements=['counterparty'],
        #                                             set_=dict(balance=15),
        #                                             where=(ModALBaseCustBal.counterparty == ins.excluded.counterparty))
        # upd_on_conflict = ins.on_conflict_do_update(constraint='unique_key',
        #                                             set_={'balance': ins.excluded.balance})

        # conn = engine.connect()
        # conn.execute(upd_on_conflict)
        # print(upd_on_conflict.returning(ModALBaseCustBal.balance))
        # conn.close()

    # check constraint
    # inspector = inspect(engine)
    # print(inspector.get_unique_constraints('customers_bal_model'))