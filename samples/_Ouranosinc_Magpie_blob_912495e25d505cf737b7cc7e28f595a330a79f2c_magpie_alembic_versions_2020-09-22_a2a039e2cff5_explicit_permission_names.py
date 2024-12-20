"""Update permission names to explicitly define applicable access and scope.

Revision ID: a2a039e2cff5
Revises: b739afcc91db
Create Date: 2020-09-22 14:35:04.540505

"""

import os
import sys

from alembic import op
from alembic.context import get_context  # noqa: F401
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.orm.session import sessionmaker
from ziggurat_foundations.models.services.group_resource_permission import GroupResourcePermissionService
from ziggurat_foundations.models.services.user_resource_permission import UserResourcePermissionService

# revision identifiers, used by Alembic.
revision = "a2a039e2cff5"
down_revision = "b739afcc91db"
branch_labels = None
depends_on = None

Session = sessionmaker()

cur_file = os.path.abspath(__file__)
root_dir = os.path.dirname(cur_file)  # version
root_dir = os.path.dirname(root_dir)  # alembic
root_dir = os.path.dirname(root_dir)  # magpie
root_dir = os.path.dirname(root_dir)  # root
sys.path.insert(0, root_dir)

from magpie.permissions import PermissionSet, Scope  # isort:skip # noqa: E402


def upgrade():
    context = get_context()
    session = Session(bind=op.get_bind())
    if not isinstance(context.connection.engine.dialect, PGDialect):
        return

    grp_res_perms = GroupResourcePermissionService.base_query(db_session=session)
    usr_res_perms = UserResourcePermissionService.base_query(db_session=session)

    for perm_list in [grp_res_perms, usr_res_perms]:
        for perm in perm_list:
            perm_name_raw = perm.perm_name
            perm_scope = Scope.RECURSIVE
            if perm_name_raw.endswith("-" + Scope.MATCH.value):
                perm_name_raw = perm_name_raw.rsplit("-", 1)[0]
                perm_scope = Scope.MATCH
            perm.perm_name = str(PermissionSet(perm_name_raw, scope=perm_scope))
    session.commit()


def downgrade():
    context = get_context()
    session = Session(bind=op.get_bind())
    if not isinstance(context.connection.engine.dialect, PGDialect):
        return

    # two following lines avoids double "DELETE" erroneous call (ignore duplicate)
    # https://stackoverflow.com/questions/28824401
    context.connection.engine.dialect.supports_sane_rowcount = False
    context.connection.engine.dialect.supports_sane_multi_rowcount = False

    grp_res_perms = GroupResourcePermissionService.base_query(db_session=session)
    usr_res_perms = UserResourcePermissionService.base_query(db_session=session)

    for perm_list in [grp_res_perms, usr_res_perms]:
        for perm in perm_list:
            perm_set = PermissionSet(perm.perm_name)
            if perm_set.implicit_permission is None:
                session.delete(perm)
            else:
                perm.perm_name = perm_set.implicit_permission
    session.commit()
