/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.anarres.jarjar.testdata.pkg2;

import org.anarres.jarjar.testdata.pkg3.Cls3;

/**
 *
 * @author shevek
 */
public class Cls2 {

    public void m_d() {
        Cls3 c = new Cls3();
        c.m_d();
    }

    public static void m_s() {
        Cls3.m_s();
    }
}
