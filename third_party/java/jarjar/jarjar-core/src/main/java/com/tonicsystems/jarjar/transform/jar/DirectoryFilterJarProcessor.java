/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.tonicsystems.jarjar.transform.jar;

/**
 *
 * @author shevek
 */
public class DirectoryFilterJarProcessor extends AbstractFilterJarProcessor {

    @Override
    protected boolean isFiltered(String name) {
        return name.endsWith("/");
    }

    @Override
    protected boolean isVerbose() {
        return false;
    }

}
