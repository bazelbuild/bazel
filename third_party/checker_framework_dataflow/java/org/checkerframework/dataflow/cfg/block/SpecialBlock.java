package org.checkerframework.dataflow.cfg.block;

/**
 * Represents a special basic block; i.e., one of the following:
 * <ul>
 * <li>Entry block of a method.</li>
 * <li>Regular exit block of a method.</li>
 * <li>Exceptional exit block of a method.</li>
 * </ul>
 *
 * @author Stefan Heule
 *
 */
public interface SpecialBlock extends SingleSuccessorBlock {

    /** The types of special basic blocks */
    public static enum SpecialBlockType {

        /** The entry block of a method */
        ENTRY,

        /** The exit block of a method */
        EXIT,

        /** A special exit block of a method for exceptional termination */
        EXCEPTIONAL_EXIT,
    }

    /**
     * @return the type of this special basic block
     */
    SpecialBlockType getSpecialType();

}
