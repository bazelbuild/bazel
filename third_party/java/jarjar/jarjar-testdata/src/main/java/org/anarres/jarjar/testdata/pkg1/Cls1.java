/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.anarres.jarjar.testdata.pkg1;

import org.anarres.jarjar.testdata.pkg2.Cls2;

/**
 *
 * @author shevek
 */
public class Cls1 {

    public void m_d() {
        Cls2 c = new Cls2();
        c.m_d();
    }

    public static void m_s() {
        Cls2.m_s();
    }
}
