/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.tonicsystems.jarjar.transform.asm;

import javax.annotation.Nonnull;
import org.objectweb.asm.ClassVisitor;

/**
 *
 * @author shevek
 */
public interface ClassTransformer {

    @Nonnull
    public ClassVisitor transform(@Nonnull ClassVisitor v);
}
