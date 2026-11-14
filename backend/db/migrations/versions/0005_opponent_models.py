"""player statistics and opponent models

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-20 12:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0005'
down_revision: Union[str, Sequence[str], None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('player_statistics',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('player_id', sa.Integer(), nullable=False),
    sa.Column('stat', sa.String(length=48), nullable=False),
    sa.Column('position', sa.String(length=8), nullable=False),
    sa.Column('street', sa.String(length=8), nullable=False),
    sa.Column('n_opportunities', sa.Integer(), nullable=False),
    sa.Column('n_hits', sa.Integer(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['player_id'], ['players.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('player_id', 'stat', 'position', 'street')
    )
    with op.batch_alter_table('player_statistics', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_player_statistics_player_id'), ['player_id'], unique=False)
    op.create_table('opponent_models',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('player_id', sa.Integer(), nullable=False),
    sa.Column('model_version', sa.String(length=32), nullable=False),
    sa.Column('posterior', sa.JSON(), nullable=False),
    sa.Column('sample_size', sa.Integer(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['player_id'], ['players.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('player_id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('opponent_models')
    with op.batch_alter_table('player_statistics', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_player_statistics_player_id'))
    op.drop_table('player_statistics')
