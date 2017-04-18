/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.anarres.jarjar.testdata.pkg0;

import org.anarres.jarjar.testdata.pkg1.Cls1;

/**
 *
 * @author shevek
 */
public class Main {

    public static void main(String[] args) {
        Cls1.m_s();
        Cls1 c = new Cls1();
        c.m_d();
    }
}
