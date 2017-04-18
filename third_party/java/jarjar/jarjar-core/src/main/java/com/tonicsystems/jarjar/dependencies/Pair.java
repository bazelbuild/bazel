/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.tonicsystems.jarjar.dependencies;

import javax.annotation.Nonnull;

/**
 *
 * @author shevek
 */
public class Pair<T> {

    private final T left;
    private final T right;

    public Pair(@Nonnull T left, @Nonnull T right) {
        this.left = left;
        this.right = right;
    }

    @Nonnull
    public T getLeft() {
        return left;
    }

    @Nonnull
    public T getRight() {
        return right;
    }

    @Override
    public int hashCode() {
        return (left.hashCode() << 8) ^ right.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (null == obj)
            return false;
        if (!getClass().equals(obj.getClass()))
            return false;
        Pair o = (Pair) obj;
        return left.equals(o.left)
                && right.equals(o.right);
    }

    @Override
    public String toString() {
        return left + " : " + right;
    }

}
