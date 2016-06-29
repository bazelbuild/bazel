package org.checkerframework.dataflow.cfg.block;

public class SpecialBlockImpl extends SingleSuccessorBlockImpl implements
        SpecialBlock {

    /** The type of this special basic block. */
    protected SpecialBlockType specialType;

    public SpecialBlockImpl(SpecialBlockType type) {
        this.specialType = type;
        this.type = BlockType.SPECIAL_BLOCK;
    }

    @Override
    public SpecialBlockType getSpecialType() {
        return specialType;
    }

    @Override
    public String toString() {
        return "SpecialBlock(" + specialType + ")";
    }

}
